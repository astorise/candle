//! Loading and dequantizing GPTQ checkpoints (AutoGPTQ / GPTQModel layout) from safetensors.
//!
//! GPTQ packs `bits`-wide integers into `i32` words and stores per-group scales and zero
//! points. This module unpacks those checkpoints into a plain dense `f32` weight matrix on
//! load so that the rest of the model can use the regular (unquantized) matmul path. There is
//! no fused/optimized (e.g. Marlin) kernel here, only a CPU dequantization step.

use candle::{DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// GPTQ quantization parameters, as found in the checkpoint's `quantize_config.json`.
#[derive(Debug, Clone, Copy)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
}

/// Unpack a GPTQ-quantized weight into a dense `[in_dim, out_dim]` `f32` tensor.
///
/// `qweight` has shape `[in_dim / pack_factor, out_dim]`, `qzeros` has shape
/// `[n_groups, out_dim / pack_factor]`, and `scales` has shape `[n_groups, out_dim]`, where
/// `pack_factor = 32 / bits` and `n_groups = ceil(in_dim / group_size)`. `g_idx`, when present,
/// maps each input row to its group index (used by "desc_act" checkpoints); otherwise rows are
/// assigned to groups sequentially based on `group_size`.
pub fn dequantize_gptq(
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: Option<&Tensor>,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    if 32 % bits != 0 {
        candle::bail!(
            "gptq: unsupported bits {bits}, only values dividing 32 (2/4/8) are supported"
        )
    }
    let pack_factor = 32 / bits;
    let mask = (1i32 << bits) - 1;

    let (packed_in, out_dim) = qweight.dims2()?;
    let in_dim = packed_in * pack_factor;
    let qweight = qweight.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let qzeros = qzeros.to_dtype(DType::I32)?.to_vec2::<i32>()?;
    let scales = scales.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let n_groups = scales.len();
    let g_idx: Vec<i32> = match g_idx {
        Some(g_idx) => g_idx.to_dtype(DType::I32)?.to_vec1::<i32>()?,
        None => (0..in_dim as i32).map(|i| i / group_size as i32).collect(),
    };

    // Unpack the per-(group, out-column) zero points.
    let mut zeros = vec![0i32; n_groups * out_dim];
    for (g, qzeros_row) in qzeros.iter().enumerate() {
        for col in 0..out_dim {
            let packed = qzeros_row[col / pack_factor];
            let shift = (col % pack_factor) * bits;
            zeros[g * out_dim + col] = ((packed >> shift) & mask) + 1;
        }
    }

    let mut weight = vec![0f32; in_dim * out_dim];
    for (row_packed, qweight_row) in qweight.iter().enumerate() {
        for col in 0..out_dim {
            let packed = qweight_row[col];
            for sub in 0..pack_factor {
                let row = row_packed * pack_factor + sub;
                if row >= in_dim {
                    break;
                }
                let shift = sub * bits;
                let q = (packed >> shift) & mask;
                let g = g_idx[row] as usize;
                let z = zeros[g * out_dim + col];
                let s = scales[g][col];
                weight[row * out_dim + col] = (q - z) as f32 * s;
            }
        }
    }
    // `to_vec2`/`to_vec1` above moved the data off whichever device the checkpoint tensors live
    // on, so the dequantized result is built on the CPU and the caller moves it back if needed.
    Tensor::from_vec(weight, (in_dim, out_dim), &candle::Device::Cpu)
}

/// Build a [`candle_nn::Linear`] layer from a GPTQ-quantized checkpoint, reading
/// `qweight`/`qzeros`/`scales`/(optional `g_idx`)/(optional `bias`) tensors at the current
/// `VarBuilder` path.
pub fn gptq_linear(
    in_dim: usize,
    out_dim: usize,
    cfg: GptqConfig,
    bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    let pack_factor = 32 / cfg.bits;
    let n_groups = in_dim.div_ceil(cfg.group_size);

    let qweight = vb.get_with_hints_dtype(
        (in_dim / pack_factor, out_dim),
        "qweight",
        Default::default(),
        DType::I32,
    )?;
    let qzeros = vb.get_with_hints_dtype(
        (n_groups, out_dim / pack_factor),
        "qzeros",
        Default::default(),
        DType::I32,
    )?;
    let scales = vb.get_with_hints_dtype(
        (n_groups, out_dim),
        "scales",
        Default::default(),
        DType::F32,
    )?;
    let g_idx = if vb.contains_tensor("g_idx") {
        Some(vb.get_with_hints_dtype(in_dim, "g_idx", Default::default(), DType::I32)?)
    } else {
        None
    };

    let weight = dequantize_gptq(
        &qweight,
        &qzeros,
        &scales,
        g_idx.as_ref(),
        cfg.bits,
        cfg.group_size,
    )?;
    let weight = weight.t()?.contiguous()?.to_device(vb.device())?;
    let bias = if bias {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    // Build a 4-bit GPTQ tensor set for a single [in_dim=8, out_dim=8] weight with one group
    // and check that dequantization recovers the values used to construct it. `out_dim` must be
    // a multiple of `pack_factor` (8 for 4-bit) for `qzeros` packing to be well-defined.
    #[test]
    fn dequantize_gptq_4bit_roundtrip() -> Result<()> {
        let bits = 4;
        let group_size = 8;
        let in_dim = 8;
        let out_dim = 8;
        let pack_factor = 32 / bits;

        // Raw quantized levels (0..15) per [row, col], chosen arbitrarily.
        let q: [[i32; 8]; 8] =
            std::array::from_fn(|row| std::array::from_fn(|col| ((row * 3 + col * 5) % 16) as i32));
        let zero_point: [i32; 8] = std::array::from_fn(|col| 1 + (col % 4) as i32);
        let scale: [f32; 8] = std::array::from_fn(|col| 0.25 * (1 + col as i32) as f32);

        // Pack qweight: [in_dim / pack_factor, out_dim].
        let mut qweight_data = vec![0i32; (in_dim / pack_factor) * out_dim];
        for col in 0..out_dim {
            let mut packed = 0i32;
            for row in 0..in_dim {
                packed |= q[row][col] << (row * bits);
            }
            qweight_data[col] = packed;
        }
        let qweight =
            Tensor::from_vec(qweight_data, (in_dim / pack_factor, out_dim), &Device::Cpu)?;

        // Pack qzeros: [n_groups=1, out_dim / pack_factor]. Stored value is `zero_point - 1`.
        let mut qzeros_data = vec![0i32; out_dim / pack_factor];
        for col in 0..out_dim {
            qzeros_data[col / pack_factor] |= (zero_point[col] - 1) << ((col % pack_factor) * bits);
        }
        let qzeros = Tensor::from_vec(qzeros_data, (1, out_dim / pack_factor), &Device::Cpu)?;

        let scales = Tensor::from_vec(scale.to_vec(), (1, out_dim), &Device::Cpu)?;

        let weight = dequantize_gptq(&qweight, &qzeros, &scales, None, bits, group_size)?;
        let weight = weight.to_vec2::<f32>()?;

        for row in 0..in_dim {
            for col in 0..out_dim {
                let expected = (q[row][col] - zero_point[col]) as f32 * scale[col];
                assert!((weight[row][col] - expected).abs() < 1e-6);
            }
        }
        Ok(())
    }
}
