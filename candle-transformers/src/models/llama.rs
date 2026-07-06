//! Llama inference implementation.
//!
//! See ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/abs/2302.13971)
//!
//! Implementation based on Hugging Face's [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }
}

/// Physical KV storage and per-sequence block table for paged attention.
///
/// Constructed and owned by the caller (e.g. a downstream paged-attention scheduler); the model
/// only reads/writes it during `forward` through the seam below. One instance is needed per
/// transformer layer, since each layer's KV storage is independent.
#[derive(Debug, Clone)]
pub struct PagedKvCache {
    /// `(num_blocks, page_block_size, num_kv_heads, head_dim)`, dtype matching the model.
    pub key_cache: Tensor,
    /// `(num_blocks, page_block_size, num_kv_heads, head_dim)`, dtype matching the model.
    pub value_cache: Tensor,
    /// `(batch_size, max_blocks)` physical block indices per sequence, dtype `U32`.
    pub block_table: Tensor,
    /// `(batch_size + 1,)` cumulative KV lengths per sequence, dtype `U32`.
    pub seqlens_k: Tensor,
    pub page_block_size: usize,
}

#[derive(Debug, Clone)]
enum KvCache {
    Contiguous(Vec<Option<(Tensor, Tensor)>>),
    Paged(Vec<PagedKvCache>),
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    kv_cache: KvCache,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

fn rope_cos_sin(dtype: DType, config: &Config, device: &Device) -> Result<(Tensor, Tensor)> {
    // precompute freqs_cis
    let theta = match &config.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => calculate_default_inv_freq(config),
        Some(rope_scaling) => {
            let low_freq_wavelen =
                rope_scaling.original_max_position_embeddings as f32 / rope_scaling.low_freq_factor;
            let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                / rope_scaling.high_freq_factor;

            calculate_default_inv_freq(config)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / rope_scaling.factor
                    } else {
                        let smooth = (rope_scaling.original_max_position_embeddings as f32
                            / wavelen
                            - rope_scaling.low_freq_factor)
                            / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                        (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                    }
                })
                .collect::<Vec<_>>()
        }
    };

    let theta = Tensor::new(theta, device)?;

    let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((config.max_position_embeddings, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    // This is different from the paper, see:
    // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
    let cos = idx_theta.cos()?.to_dtype(dtype)?;
    let sin = idx_theta.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let (cos, sin) = rope_cos_sin(dtype, config, device)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kv_cache: KvCache::Contiguous(vec![None; config.num_hidden_layers]),
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Build a cache whose KV storage is caller-owned paged blocks (one [`PagedKvCache`] per
    /// transformer layer) instead of the contiguous concat-and-narrow storage `new` uses.
    ///
    /// Rotary embeddings and causal-mask caching are shared infrastructure and behave exactly as
    /// they do for the contiguous cache; only KV storage and the attention kernel used in
    /// `CausalSelfAttention::forward` differ. Requires the `flash-attn` feature at call time.
    pub fn new_paged(
        dtype: DType,
        config: &Config,
        device: &Device,
        paged_kvs: Vec<PagedKvCache>,
    ) -> Result<Self> {
        if paged_kvs.len() != config.num_hidden_layers {
            candle::bail!(
                "new_paged: expected {} paged kv caches (one per layer), got {}",
                config.num_hidden_layers,
                paged_kvs.len()
            )
        }
        let (cos, sin) = rope_cos_sin(dtype, config, device)?;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache: true,
            kv_cache: KvCache::Paged(paged_kvs),
            device: device.clone(),
            cos,
            sin,
        })
    }
}

fn causal_mask(
    masks: &mut HashMap<(usize, usize), Tensor>,
    device: &Device,
    seq_len: usize,
    index_pos: usize,
) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    if let Some(mask) = masks.get(&(seq_len, kv_len)) {
        Ok(mask.clone())
    } else {
        let mask = crate::utils::build_causal_mask(seq_len, index_pos, device)?;
        masks.insert((seq_len, kv_len), mask.clone());
        Ok(mask)
    }
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
    max_position_embeddings: usize,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        let y = match &mut cache.kv_cache {
            KvCache::Paged(paged_layers) => {
                self.forward_paged(&q, &k, &v, index_pos, &mut paged_layers[block_idx])?
            }
            KvCache::Contiguous(kvs) => self.forward_contiguous(
                q,
                k,
                v,
                index_pos,
                cache.use_kv_cache,
                &mut kvs[block_idx],
                &mut cache.masks,
                &cache.device,
                b_sz,
                seq_len,
                hidden_size,
            )?,
        };
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_contiguous(
        &self,
        q: Tensor,
        mut k: Tensor,
        mut v: Tensor,
        index_pos: usize,
        use_kv_cache: bool,
        kv: &mut Option<(Tensor, Tensor)>,
        masks: &mut HashMap<(usize, usize), Tensor>,
        device: &Device,
        b_sz: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor> {
        if use_kv_cache {
            if let Some((cache_k, cache_v)) = kv.as_ref() {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?
                }
            }
            *kv = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask =
                    causal_mask(masks, device, seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        Ok(y)
    }

    #[cfg(feature = "flash-attn")]
    fn forward_paged(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        index_pos: usize,
        paged: &mut PagedKvCache,
    ) -> Result<Tensor> {
        // q/k/v come in as (b_sz, heads, seq_len, head_dim); paged storage and the varlen kernel
        // both expect a heads-last layout of (.., seq_len, heads, head_dim).
        let (b_sz, _, seq_len, _) = q.dims4()?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let device = q.device();

        let (num_blocks, page_block_size, num_kv_heads, head_dim) = paged.key_cache.dims4()?;
        let block_table: Vec<Vec<u32>> = paged.block_table.to_vec2()?;

        let mut slot_ids = Vec::with_capacity(b_sz * seq_len);
        for row in block_table.iter().take(b_sz) {
            for i in 0..seq_len {
                let pos = index_pos + i;
                let logical_block = pos / page_block_size;
                let offset = (pos % page_block_size) as u32;
                let block_id = *row.get(logical_block).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "paged kv cache: block_table has no entry for logical block {logical_block}"
                    ))
                })?;
                slot_ids.push(block_id * page_block_size as u32 + offset);
            }
        }
        let slot_ids = Tensor::new(slot_ids, device)?.reshape((b_sz * seq_len, 1, 1))?;

        let k_new = k.reshape((b_sz * seq_len, num_kv_heads, head_dim))?;
        let v_new = v.reshape((b_sz * seq_len, num_kv_heads, head_dim))?;
        let idx = slot_ids
            .broadcast_as((b_sz * seq_len, num_kv_heads, head_dim))?
            .contiguous()?;

        let key_cache_flat =
            paged
                .key_cache
                .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?;
        key_cache_flat.scatter_set(&idx, &k_new, 0)?;
        let value_cache_flat =
            paged
                .value_cache
                .reshape((num_blocks * page_block_size, num_kv_heads, head_dim))?;
        value_cache_flat.scatter_set(&idx, &v_new, 0)?;

        let seqlens_k: Vec<u32> = paged.seqlens_k.to_vec1()?;
        let max_seqlen_k = seqlens_k.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0) as usize;
        let seqlens_q: Vec<u32> = (0..=b_sz as u32).map(|i| i * seq_len as u32).collect();
        let seqlens_q = Tensor::new(seqlens_q, device)?;

        let q_flat = q.reshape((b_sz * seq_len, self.num_attention_heads, self.head_dim))?;
        let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
        let y = candle_flash_attn::flash_attn_varlen_paged_windowed(
            &q_flat,
            &paged.key_cache,
            &paged.value_cache,
            &seqlens_q,
            &paged.seqlens_k,
            &paged.block_table,
            None,
            seq_len,
            max_seqlen_k,
            softmax_scale,
            None,
            None,
            page_block_size,
            None,
        )?;
        y.reshape((b_sz, seq_len, self.num_attention_heads * self.head_dim))
    }

    #[cfg(not(feature = "flash-attn"))]
    fn forward_paged(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _index_pos: usize,
        _paged: &mut PagedKvCache,
    ) -> Result<Tensor> {
        candle::bail!("paged kv-cache attention requires the 'flash-attn' feature")
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        crate::utils::repeat_kv(x, self.num_attention_heads / self.num_key_value_heads)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[derive(Debug, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    // required by LLaVA
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }
    // required by LLaVA
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(wte.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        Config {
            hidden_size: 8,
            intermediate_size: 16,
            vocab_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn cache_new_is_contiguous_by_default() -> Result<()> {
        let device = Device::Cpu;
        let cache = Cache::new(true, DType::F32, &tiny_config(), &device)?;
        assert!(matches!(cache.kv_cache, KvCache::Contiguous(_)));
        Ok(())
    }

    #[test]
    fn new_paged_rejects_layer_count_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();
        // Only one PagedKvCache for a 2-layer config: must be rejected up front rather than
        // panicking on out-of-bounds access during forward.
        let paged = PagedKvCache {
            key_cache: Tensor::zeros((1, 4, cfg.num_key_value_heads, 4), DType::F32, &device)?,
            value_cache: Tensor::zeros((1, 4, cfg.num_key_value_heads, 4), DType::F32, &device)?,
            block_table: Tensor::zeros((1, 1), DType::U32, &device)?,
            seqlens_k: Tensor::zeros((2,), DType::U32, &device)?,
            page_block_size: 4,
        };
        let err = Cache::new_paged(DType::F32, &cfg, &device, vec![paged]);
        assert!(err.is_err());
        Ok(())
    }
}
