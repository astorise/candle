//! Fused dequantize+GEMM CUDA kernel for block-wise FP8 (E4M3) quantized linear layers
//! (DeepSeek-V3 style checkpoints).
//!
//! Unlike `candle_transformers::quantized_fp8`, which dequantizes a block-wise FP8 checkpoint
//! into a dense `f32` weight once at load time and then runs the regular matmul, this crate
//! fuses the per-element dequantization into a shared-memory-tiled GEMM kernel so the dense
//! weight is never materialized. The GEMM itself is a straightforward shared-memory-tiled
//! kernel with scalar FP32 accumulation; it does not use tensor cores.

mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Tensor};

pub struct Fp8BlockGemm {
    pub block_size: usize,
}

impl candle::CustomOp3 for Fp8BlockGemm {
    fn name(&self) -> &'static str {
        "fp8-block-gemm"
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
            "no cpu support for the fused fp8-block-gemm kernel, use candle_transformers::quantized_fp8 instead"
        )
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        w: &CudaStorage,
        w_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if x.dtype() != DType::F32 {
            candle::bail!(
                "fp8-block-gemm only supports f32 activations, got {:?}",
                x.dtype()
            );
        }
        if w.dtype() != DType::F8E4M3 {
            candle::bail!(
                "fp8-block-gemm expects an F8E4M3 weight, got {:?}",
                w.dtype()
            );
        }
        if scale.dtype() != DType::F32 {
            candle::bail!(
                "fp8-block-gemm expects an f32 scale, got {:?}",
                scale.dtype()
            );
        }

        let (m, k) = x_l.shape().dims2()?;
        let (n, wk) = w_l.shape().dims2()?;
        if wk != k {
            candle::bail!("fp8-block-gemm: weight cols {wk} != x cols {k}");
        }
        let (scale_rows, scale_cols) = scale_l.shape().dims2()?;
        if scale_rows != n.div_ceil(self.block_size) || scale_cols != k.div_ceil(self.block_size) {
            candle::bail!(
                "fp8-block-gemm: scale shape {:?} does not match weight shape {:?} for block_size {}",
                (scale_rows, scale_cols),
                (n, k),
                self.block_size
            );
        }

        let dev = x.device();
        let stream = dev.cuda_stream();

        let x_slice = x.as_cuda_slice::<f32>()?;
        let x_slice = match x_l.contiguous_offsets() {
            Some((o1, o2)) => x_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: x must be contiguous"),
        };
        let w_slice = w.as_cuda_slice::<float8::F8E4M3>()?;
        let w_slice = match w_l.contiguous_offsets() {
            Some((o1, o2)) => w_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: weight must be contiguous"),
        };
        let scale_slice = scale.as_cuda_slice::<f32>()?;
        let scale_slice = match scale_l.contiguous_offsets() {
            Some((o1, o2)) => scale_slice.slice(o1..o2),
            None => candle::bail!("fp8-block-gemm: scale must be contiguous"),
        };

        let mut dst = unsafe { dev.alloc::<f32>(m * n)? };

        unsafe {
            let (x_ptr, _guard) = x_slice.device_ptr(&stream);
            let (w_ptr, _guard) = w_slice.device_ptr(&stream);
            let (scale_ptr, _guard) = scale_slice.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            ffi::run_fp8_block_gemm_f32(
                x_ptr as *const f32,
                w_ptr as *const core::ffi::c_void,
                scale_ptr as *const f32,
                dst_ptr as *mut f32,
                m as i32,
                k as i32,
                n as i32,
                self.block_size as i32,
                scale_cols as i32,
            );
        }

        let dst = CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, Shape::from((m, n))))
    }
}

/// Run the fused block-wise FP8 dequant+GEMM kernel: `y = x @ dequant(weight, scale)^T`.
///
/// * `x` - activations, `f32`, shape `[m, k]`.
/// * `weight` - `F8E4M3` weight, shape `[n, k]` (standard `nn.Linear` layout).
/// * `scale` - per-block `f32` scale, shape `[ceil(n / block_size), ceil(k / block_size)]`.
pub fn fp8_block_gemm(
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    block_size: usize,
) -> Result<Tensor> {
    let op = Fp8BlockGemm { block_size };
    x.apply_op3(weight, scale, op)
}
