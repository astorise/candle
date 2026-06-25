//! Correctness tests for the block-paged KV-cache attention reference.
//!
//! The paged implementation is validated against a straightforward dense
//! attention computed over the same keys/values, for both multi-head and
//! grouped-query configurations, and across multiple block sizes / contexts.

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::attention::paged::{paged_attention, reshape_and_cache, BlockAllocator};
use candle_nn::ops::softmax_last_dim;

const EPS: f32 = 1e-4;

/// Dense single-token attention reference for one sequence.
/// q: [num_heads, head_dim], k/v: [ctx, num_kv_heads, head_dim].
fn dense_decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    num_kv_heads: usize,
) -> Result<Tensor> {
    let (num_heads, head_dim) = q.dims2()?;
    let (ctx, _, _) = k.dims3()?;
    let group = num_heads / num_kv_heads;

    let expand = |t: &Tensor| -> Result<Tensor> {
        t.reshape((ctx, num_kv_heads, 1, head_dim))?
            .broadcast_as((ctx, num_kv_heads, group, head_dim))?
            .reshape((ctx, num_heads, head_dim))
    };
    let k = expand(k)?.transpose(0, 1)?.contiguous()?; // [H, ctx, D]
    let v = expand(v)?.transpose(0, 1)?.contiguous()?; // [H, ctx, D]
    let q = q.unsqueeze(1)?; // [H, 1, D]

    let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale as f64)?; // [H,1,ctx]
    let probs = softmax_last_dim(&scores)?;
    let out = probs.matmul(&v)?; // [H,1,D]
    out.squeeze(1)
}

/// Build a paged cache by laying each sequence's KV out across freshly
/// allocated blocks, returning the cache tensors plus the per-batch
/// block_tables / context_lens needed by `paged_attention`.
#[allow(clippy::type_complexity)]
fn build_paged_cache(
    keys: &[Tensor], // each [ctx_i, num_kv_heads, head_dim]
    values: &[Tensor],
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mut allocator = BlockAllocator::new(num_blocks, block_size);
    let mut k_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        device,
    )?;
    let mut v_cache = k_cache.clone();

    let mut block_tables: Vec<Vec<u32>> = Vec::new();
    let mut context_lens: Vec<u32> = Vec::new();
    let mut max_blocks = 1usize;

    for (k, v) in keys.iter().zip(values.iter()) {
        let ctx = k.dim(0)?;
        let n_blocks = ctx.div_ceil(block_size);
        let mut table = Vec::with_capacity(n_blocks);
        let mut slots = Vec::with_capacity(ctx);
        for b in 0..n_blocks {
            let phys = allocator.allocate().expect("pool exhausted");
            table.push(phys as u32);
            let tokens_in_block = (ctx - b * block_size).min(block_size);
            for off in 0..tokens_in_block {
                slots.push((phys * block_size + off) as u32);
            }
        }
        let slot_mapping = Tensor::from_vec(slots, ctx, device)?;
        let (kc, vc) = reshape_and_cache(k, v, &k_cache, &v_cache, &slot_mapping)?;
        k_cache = kc;
        v_cache = vc;

        max_blocks = max_blocks.max(n_blocks);
        block_tables.push(table);
        context_lens.push(ctx as u32);
    }

    // Pad block tables to a rectangular [num_seqs, max_blocks].
    for table in block_tables.iter_mut() {
        table.resize(max_blocks, 0);
    }
    let flat: Vec<u32> = block_tables.iter().flatten().copied().collect();
    let block_tables = Tensor::from_vec(flat, (keys.len(), max_blocks), device)?;
    let context_lens = Tensor::from_vec(context_lens, keys.len(), device)?;

    Ok((k_cache, v_cache, block_tables, context_lens))
}

fn run_case(
    num_kv_heads: usize,
    group: usize,
    head_dim: usize,
    block_size: usize,
    ctx_lens: &[usize],
) -> Result<()> {
    run_case_on(
        &Device::Cpu,
        num_kv_heads,
        group,
        head_dim,
        block_size,
        ctx_lens,
    )
}

fn run_case_on(
    device: &Device,
    num_kv_heads: usize,
    group: usize,
    head_dim: usize,
    block_size: usize,
    ctx_lens: &[usize],
) -> Result<()> {
    let num_heads = num_kv_heads * group;
    let num_seqs = ctx_lens.len();
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Deterministic pseudo-random inputs (no rng dependency).
    let gen = |n: usize, seed: f32| -> Result<Tensor> {
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.123 + seed).sin()) * 0.5)
            .collect();
        Tensor::from_vec(data, n, device)
    };

    let q = gen(num_seqs * num_heads * head_dim, 0.0)?.reshape((num_seqs, num_heads, head_dim))?;

    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (s, &ctx) in ctx_lens.iter().enumerate() {
        keys.push(
            gen(ctx * num_kv_heads * head_dim, 1.0 + s as f32)?.reshape((
                ctx,
                num_kv_heads,
                head_dim,
            ))?,
        );
        values.push(
            gen(ctx * num_kv_heads * head_dim, 7.0 + s as f32)?.reshape((
                ctx,
                num_kv_heads,
                head_dim,
            ))?,
        );
    }

    let total_blocks: usize = ctx_lens.iter().map(|c| c.div_ceil(block_size)).sum();
    let num_blocks = total_blocks + 2; // a little slack in the pool

    let (k_cache, v_cache, block_tables, context_lens) = build_paged_cache(
        &keys,
        &values,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device,
    )?;

    let out = paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        block_size,
        scale,
        None,
    )?;
    assert_eq!(out.dims(), &[num_seqs, num_heads, head_dim]);

    for s in 0..num_seqs {
        let expected = dense_decode(&q.i(s)?, &keys[s], &values[s], scale, num_kv_heads)?;
        let got = out.i(s)?;
        let diff = (got - expected)?
            .abs()?
            .max(0)?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < EPS,
            "seq {s}: max abs diff {diff} exceeds {EPS} (kv_heads={num_kv_heads}, group={group}, block_size={block_size}, ctx={})",
            ctx_lens[s]
        );
    }
    Ok(())
}

#[test]
fn paged_matches_dense_mha() -> Result<()> {
    // Multi-head (group = 1), contexts that do and do not fill whole blocks.
    run_case(4, 1, 8, 4, &[4, 7, 1, 16])
}

#[test]
fn paged_matches_dense_gqa() -> Result<()> {
    // Grouped-query attention: 2 KV heads shared by 8 query heads.
    run_case(2, 4, 16, 8, &[10, 3, 20])
}

#[test]
fn paged_matches_dense_mqa_block_size_one() -> Result<()> {
    // Multi-query attention with the degenerate block size of 1.
    run_case(1, 8, 8, 1, &[5, 9])
}

#[test]
fn empty_context_yields_zeros() -> Result<()> {
    let device = Device::Cpu;
    let (num_blocks, block_size, num_kv_heads, head_dim) = (4, 4, 2, 8);
    let num_heads = 2;
    let k_cache = Tensor::zeros(
        (num_blocks, block_size, num_kv_heads, head_dim),
        DType::F32,
        &device,
    )?;
    let v_cache = k_cache.clone();
    let q = Tensor::ones((1, num_heads, head_dim), DType::F32, &device)?;
    let block_tables = Tensor::zeros((1, 1), DType::U32, &device)?;
    let context_lens = Tensor::zeros(1, DType::U32, &device)?;
    let out = paged_attention(
        &q,
        &k_cache,
        &v_cache,
        &block_tables,
        &context_lens,
        block_size,
        0.5,
        None,
    )?;
    let sum = out.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert_eq!(sum, 0.0);
    Ok(())
}

#[test]
fn block_allocator_pools_blocks() {
    let mut alloc = BlockAllocator::new(3, 8);
    assert_eq!(alloc.num_free(), 3);
    let a = alloc.allocate().unwrap();
    let b = alloc.allocate().unwrap();
    let c = alloc.allocate().unwrap();
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_eq!(alloc.allocate(), None);
    alloc.free(b);
    assert_eq!(alloc.num_free(), 1);
    assert_eq!(alloc.allocate(), Some(b));
}

// ------------------------------------------------------------------------
// CUDA path. These exercise `paged_attention` / `reshape_and_cache` on an
// actual GPU device and are run by the `ci_cuda` GPU runner workflow
// (`cargo test --features cuda`). They are compiled out on non-CUDA builds.
// ------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device 0 must be available on the GPU runner")
    }

    #[test]
    fn paged_matches_dense_mha_cuda() -> Result<()> {
        run_case_on(&cuda_device(), 4, 1, 8, 4, &[4, 7, 1, 16])
    }

    #[test]
    fn paged_matches_dense_gqa_cuda() -> Result<()> {
        run_case_on(&cuda_device(), 2, 4, 16, 8, &[10, 3, 20])
    }

    #[test]
    fn paged_matches_dense_mqa_block_size_one_cuda() -> Result<()> {
        run_case_on(&cuda_device(), 1, 8, 8, 1, &[5, 9])
    }

    /// The implementation is device-agnostic, so the CUDA result must match
    /// the CPU result bit-for-tolerance, not just the dense reference.
    #[test]
    fn paged_cuda_matches_cpu() -> Result<()> {
        let (num_kv_heads, group, head_dim, block_size) = (2, 4, 16, 8);
        let num_heads = num_kv_heads * group;
        let ctx_lens = [10usize, 3, 20];
        let num_seqs = ctx_lens.len();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let gen = |dev: &Device, n: usize, seed: f32| -> Result<Tensor> {
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.123 + seed).sin()) * 0.5)
                .collect();
            Tensor::from_vec(data, n, dev)
        };

        let run = |dev: &Device| -> Result<Tensor> {
            let q = gen(dev, num_seqs * num_heads * head_dim, 0.0)?
                .reshape((num_seqs, num_heads, head_dim))?;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            for (s, &ctx) in ctx_lens.iter().enumerate() {
                keys.push(
                    gen(dev, ctx * num_kv_heads * head_dim, 1.0 + s as f32)?.reshape((
                        ctx,
                        num_kv_heads,
                        head_dim,
                    ))?,
                );
                values.push(
                    gen(dev, ctx * num_kv_heads * head_dim, 7.0 + s as f32)?.reshape((
                        ctx,
                        num_kv_heads,
                        head_dim,
                    ))?,
                );
            }
            let total_blocks: usize = ctx_lens.iter().map(|c| c.div_ceil(block_size)).sum();
            let (k_cache, v_cache, block_tables, context_lens) = build_paged_cache(
                &keys,
                &values,
                total_blocks + 2,
                block_size,
                num_kv_heads,
                head_dim,
                dev,
            )?;
            paged_attention(
                &q,
                &k_cache,
                &v_cache,
                &block_tables,
                &context_lens,
                block_size,
                scale,
                None,
            )
        };

        let cpu = run(&Device::Cpu)?;
        let gpu = run(&cuda_device())?.to_device(&Device::Cpu)?;
        let diff = (cpu - gpu)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < EPS,
            "CUDA vs CPU paged attention diff {diff} exceeds {EPS}"
        );
        Ok(())
    }
}
