//! Fused dequantize+GEMM CUDA kernel for AWQ (AutoAWQ "GEMM" layout, 4-bit) quantized linear
//! layers.
//!
//! Unlike `candle_transformers::quantized_awq`, which dequantizes an AWQ checkpoint into a dense
//! `f32` weight once at load time and then runs the regular matmul, this crate fuses the
//! per-element dequantization (including AWQ's output-axis nibble permutation) into a
//! shared-memory-tiled GEMM kernel so the dense weight is never materialized. The GEMM itself
//! is a straightforward shared-memory-tiled kernel with scalar FP32 accumulation; it does not
//! use tensor cores.

mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};

/// `x.apply_op3(qweight, qzeros, op)`: `scales` rides along as an extra field, mirroring the
/// pattern `candle-flash-attn` uses for `alibi_slopes` (CustomOp only supports 3 tensors).
pub struct AwqGemm {
    pub scales: Tensor,
    pub group_size: usize,
}

const AWQ_BITS: usize = 4;
const AWQ_PACK_FACTOR: usize = 32 / AWQ_BITS;

impl candle::CustomOp3 for AwqGemm {
    fn name(&self) -> &'static str {
        "awq-gemm"
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
        candle::bail!(
            "no cpu support for the fused awq-gemm kernel, use candle_transformers::quantized_awq instead"
        )
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
                "awq-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        let (m, k) = x_l.shape().dims2()?;
        let (qk, n_packed) = qweight_l.shape().dims2()?;
        if qk != k {
            candle::bail!("awq-gemm: qweight rows {qk} != x cols {k}");
        }
        let n = n_packed * AWQ_PACK_FACTOR;
        let (_n_groups, qz_packed) = qzeros_l.shape().dims2()?;
        if qz_packed != n_packed {
            candle::bail!("awq-gemm: qzeros cols {qz_packed} != qweight cols {n_packed}");
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: x must be contiguous"),
        };
        let qweight_slice = qweight.as_cuda_slice::<i32>()?;
        let qweight_slice = match qweight_l.contiguous_offsets() {
            Some((o1, o2)) => qweight_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: qweight must be contiguous"),
        };
        let qzeros_slice = qzeros.as_cuda_slice::<i32>()?;
        let qzeros_slice = match qzeros_l.contiguous_offsets() {
            Some((o1, o2)) => qzeros_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: qzeros must be contiguous"),
        };

        if self.scales.dtype() != DType::F32 {
            candle::bail!(
                "awq-gemm: scales must be f32, got {:?}",
                self.scales.dtype()
            );
        }
        let (scales_storage, scales_layout) = self.scales.storage_and_layout();
        let scales_slice = match &*scales_storage {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("awq-gemm: scales must be a cuda tensor"),
        };
        let scales_slice = match scales_layout.contiguous_offsets() {
            Some((o1, o2)) => scales_slice.slice(o1..o2),
            None => candle::bail!("awq-gemm: scales must be contiguous"),
        };

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (qweight_ptr, _guard) = qweight_slice.device_ptr(&stream);
            let (qzeros_ptr, _guard) = qzeros_slice.device_ptr(&stream);
            let (scales_ptr, _guard) = scales_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_awq_gemm_f32(
                x_ptr as *const f32,
                qweight_ptr as *const i32,
                qzeros_ptr as *const i32,
                scales_ptr as *const f32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.group_size as i32,
                n_packed as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused AWQ (4-bit GEMM layout) dequant+GEMM kernel:
/// `y = x @ dequant(qweight, qzeros, scales)`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `qweight` - packed weight, `i32`, shape `[k, n / pack_factor]`.
/// * `qzeros` - packed zero points, `i32`, shape `[n_groups, n / pack_factor]`.
/// * `scales` - per-group scales, `f32`, shape `[n_groups, n]`.
pub fn awq_gemm(
    x: &Tensor,
    qweight: &Tensor,
    qzeros: &Tensor,
    scales: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let op = AwqGemm {
        scales: scales.clone(),
        group_size,
    };
    x.apply_op3(qweight, qzeros, op)
}
