//! Shared utilities: repeat_kv, repeat_penalty, causal mask.

use candle::{Device, Result, Tensor};

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Applies OpenAI-style frequency and presence penalties to `logits`, given the
/// token `context` generated so far.
///
/// - `frequency_penalty` is scaled by how many times each token already appears
///   in `context`.
/// - `presence_penalty` is applied once to every token that appears at least
///   once in `context`, regardless of its count.
pub fn apply_freq_presence_penalty(
    logits: &Tensor,
    frequency_penalty: f32,
    presence_penalty: f32,
    context: &[u32],
) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    if frequency_penalty != 0. || presence_penalty != 0. {
        let mut counts = std::collections::HashMap::new();
        for &token_id in context {
            *counts.entry(token_id).or_insert(0u32) += 1;
        }
        for (token_id, count) in counts {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit -= count as f32 * frequency_penalty + presence_penalty;
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_presence_penalty_scales_with_token_count() -> Result<()> {
        let logits = Tensor::new(&[1f32, 1f32, 1f32], &Device::Cpu)?;
        // Token 1 appears twice, token 2 once, token 0 never.
        let context = [1u32, 1, 2];
        let penalized = apply_freq_presence_penalty(&logits, 0.5, 0.2, &context)?;
        let penalized = penalized.to_vec1::<f32>()?;
        let expected = [1.0, 1.0 - 2. * 0.5 - 0.2, 1.0 - 0.5 - 0.2];
        for (got, want) in penalized.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{got} != {want}");
        }
        Ok(())
    }

    #[test]
    fn freq_presence_penalty_noop_when_zero() -> Result<()> {
        let logits = Tensor::new(&[3f32, -2f32, 0.5f32], &Device::Cpu)?;
        let penalized = apply_freq_presence_penalty(&logits, 0., 0., &[0, 0, 1])?;
        assert_eq!(penalized.to_vec1::<f32>()?, logits.to_vec1::<f32>()?);
        Ok(())
    }
}
