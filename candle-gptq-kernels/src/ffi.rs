use core::ffi::c_int;

extern "C" {
    pub(crate) fn run_gptq_gemm_f32(
        x: *const f32,
        qweight: *const i32,
        qzeros: *const i32,
        scales: *const f32,
        g_idx: *const i32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        bits: c_int,
        pack_factor: c_int,
        n_groups_out: c_int,
    );

    /// Tensor-core (WMMA `mma.sync`) variant, 4-bit only: `bits`/`pack_factor` are fixed at 4/8
    /// in the kernel itself, so they are not parameters here.
    pub(crate) fn run_gptq_gemm_tc_f32(
        x: *const f32,
        qweight: *const i32,
        qzeros: *const i32,
        scales: *const f32,
        g_idx: *const i32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        n_groups_out: c_int,
    );
}
