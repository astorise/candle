//! Fused dequantize+GEMM CUDA kernel for GPTQ (AutoGPTQ/GPTQModel "old" CUDA layout)
//! quantized linear layers.
//!
//! Unlike `candle_transformers::quantized_gptq`, which dequantizes a GPTQ checkpoint into a
//! dense `f32` weight once at load time and then runs the regular matmul, this crate fuses the
//! per-element dequantization into a shared-memory-tiled GEMM kernel so the dense weight is
//! never materialized. The GEMM itself is a straightforward shared-memory-tiled kernel with
//! scalar FP32 accumulation; it does not use tensor cores.

mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};

/// `x.apply_op3(qweight, qzeros, op)`: `scales` and `g_idx` ride along as extra fields, mirroring
/// the pattern `candle-flash-attn` uses for `alibi_slopes` (CustomOp only supports 3 tensors).
pub struct GptqGemm {
    pub scales: Tensor,
    pub g_idx: Tensor,
    pub bits: usize,
    pub pack_factor: usize,
}

impl GptqGemm {
    fn extra_cuda_slice<'a, T: candle::cuda_backend::CudaDType>(
        storage: &'a candle::Storage,
        layout: &Layout,
        name: &'static str,
    ) -> Result<candle::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let slice = match storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("{name} must be a cuda tensor"),
        };
        match layout.contiguous_offsets() {
            Some((o1, o2)) => Ok(slice.slice(o1..o2)),
            None => candle::bail!("{name} has to be contiguous"),
        }
    }
}

impl candle::CustomOp3 for GptqGemm {
    fn name(&self) -> &'static str {
        "gptq-gemm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for the fused gptq-gemm kernel, use candle_transformers::quantized_gptq instead")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        qzeros: &CudaStorage,
        qzeros_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if x.dtype() != DType::F32 {
            candle::bail!(
                "gptq-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        let (m, k) = x_l.shape().dims2()?;
        let (packed_k, n) = qweight_l.shape().dims2()?;
        if packed_k * self.pack_factor != k {
            candle::bail!(
                "gptq-gemm: qweight rows {packed_k} * pack_factor {} != x cols {k}",
                self.pack_factor
            );
        }
        let (_n_groups, n_packed) = qzeros_l.shape().dims2()?;
        if n_packed * self.pack_factor != n {
            candle::bail!(
                "gptq-gemm: qzeros cols {n_packed} * pack_factor {} != out dim {n}",
                self.pack_factor
            );
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: x must be contiguous"),
        };
        let qweight_slice = qweight.as_cuda_slice::<i32>()?;
        let qweight_slice = match qweight_l.contiguous_offsets() {
            Some((o1, o2)) => qweight_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: qweight must be contiguous"),
        };
        let qzeros_slice = qzeros.as_cuda_slice::<i32>()?;
        let qzeros_slice = match qzeros_l.contiguous_offsets() {
            Some((o1, o2)) => qzeros_slice.slice(o1..o2),
            None => candle::bail!("gptq-gemm: qzeros must be contiguous"),
        };
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let scales_slice = Self::extra_cuda_slice::<f32>(&scales_storage, scales_layout, "scales")?;
        let (g_idx_storage, g_idx_layout) = self.g_idx.storage_and_layout();
        let g_idx_slice = Self::extra_cuda_slice::<i32>(&g_idx_storage, g_idx_layout, "g_idx")?;

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (qweight_ptr, _guard) = qweight_slice.device_ptr(&stream);
            let (qzeros_ptr, _guard) = qzeros_slice.device_ptr(&stream);
            let (scales_ptr, _guard) = scales_slice.device_ptr(&stream);
            let (g_idx_ptr, _guard) = g_idx_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_gptq_gemm_f32(
                x_ptr as *const f32,
                qweight_ptr as *const i32,
                qzeros_ptr as *const i32,
                scales_ptr as *const f32,
                g_idx_ptr as *const i32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.bits as i32,
                self.pack_factor as i32,
                n_packed as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused GPTQ dequant+GEMM kernel: `y = x @ dequant(qweight, qzeros, scales, g_idx)`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `qweight` - packed weight, `i32`, shape `[k / pack_factor, n]`.
/// * `qzeros` - packed zero points, `i32`, shape `[n_groups, n / pack_factor]`.
/// * `scales` - per-group scales, `f32`, shape `[n_groups, n]`.
/// * `g_idx` - per-row group index, `i32`, shape `[k]`.
pub fn gptq_gemm(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    g_idx: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<Tensor> {
    let _ = group_size;
    let pack_factor = 32 / bits;
    let op = GptqGemm {
        scales: scales.clone(),
        g_idx: g_idx.clone(),
        bits,
        pack_factor,
    };
    x.apply_op3(qweight, qzeros, op)
}
