//! Low-Rank Adaptation (LoRA) building blocks.
//!
//! [`LoraLinear`] wraps a frozen base [`Linear`] layer and lets one or more
//! named low-rank adapters be registered against it, following the PEFT
//! convention of storing adapters as a pair of `lora_A` / `lora_B` matrices.
//! Which adapter (if any) contributes to the output is controlled through a
//! shared [`ActiveAdapter`] handle: cloning that handle into every
//! `LoraLinear` of a model lets a single call retarget the whole model to a
//! different adapter (or back to the frozen base weights) without reloading
//! anything.
use crate::{Linear, Module, VarBuilder};
use candle::{Result, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Rank and scaling factor for a LoRA adapter. Following the original paper,
/// the adapter contributes `lora_b(lora_a(x)) * (alpha / rank)` on top of the
/// frozen base layer.
#[derive(Debug, Clone, Copy)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
}

impl LoraConfig {
    pub fn new(rank: usize, alpha: f64) -> Self {
        Self { rank, alpha }
    }

    fn scaling(&self) -> f64 {
        self.alpha / self.rank as f64
    }
}

/// A shared handle selecting which adapter (if any) is currently active on a
/// group of [`LoraLinear`] layers.
#[derive(Debug, Clone, Default)]
pub struct ActiveAdapter(Rc<RefCell<Option<String>>>);

impl ActiveAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self) -> Option<String> {
        self.0.borrow().clone()
    }

    pub fn set(&self, name: Option<String>) {
        *self.0.borrow_mut() = name;
    }
}

#[derive(Debug, Clone)]
struct Adapter {
    lora_a: Linear,
    lora_b: Linear,
    scaling: f64,
}

impl Adapter {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.lora_a.forward(x)?;
        let x = self.lora_b.forward(&x)?;
        x.affine(self.scaling, 0.)
    }
}

/// A linear layer augmented with zero or more named LoRA adapters. With no
/// adapter registered, or none selected as active, it behaves exactly like
/// the wrapped base layer.
#[derive(Debug, Clone)]
pub struct LoraLinear {
    base: Linear,
    adapters: HashMap<String, Adapter>,
    active: ActiveAdapter,
}

impl LoraLinear {
    pub fn new(base: Linear, active: ActiveAdapter) -> Self {
        Self {
            base,
            adapters: HashMap::new(),
            active,
        }
    }

    pub fn base(&self) -> &Linear {
        &self.base
    }

    pub fn has_adapter(&self, name: &str) -> bool {
        self.adapters.contains_key(name)
    }

    /// Loads a PEFT-style adapter from `vb`, expecting `lora_A.weight`
    /// (shape `(rank, in_features)`) and `lora_B.weight` (shape
    /// `(out_features, rank)`) tensors, and registers it under `name`.
    pub fn load_adapter(
        &mut self,
        name: &str,
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        config: &LoraConfig,
    ) -> Result<()> {
        let lora_a = Linear::new(vb.get((config.rank, in_features), "lora_A.weight")?, None);
        let lora_b = Linear::new(vb.get((out_features, config.rank), "lora_B.weight")?, None);
        self.adapters.insert(
            name.to_string(),
            Adapter {
                lora_a,
                lora_b,
                scaling: config.scaling(),
            },
        );
        Ok(())
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.base.forward(x)?;
        let active = self.active.get();
        match active.as_deref().and_then(|name| self.adapters.get(name)) {
            Some(adapter) => &y + adapter.forward(x)?,
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarBuilder;
    use candle::{DType, Device};
    use std::collections::HashMap;

    fn adapter_vb(
        rank: usize,
        in_features: usize,
        out_features: usize,
        dev: &Device,
    ) -> VarBuilder<'static> {
        let a = Tensor::ones((rank, in_features), DType::F32, dev).unwrap();
        let b = Tensor::ones((out_features, rank), DType::F32, dev).unwrap();
        let mut ts = HashMap::new();
        ts.insert("lora_A.weight".to_string(), a);
        ts.insert("lora_B.weight".to_string(), b);
        VarBuilder::from_tensors(ts, DType::F32, dev)
    }

    #[test]
    fn no_active_adapter_matches_base() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::ones((4, 3), DType::F32, &dev)?;
        let base = Linear::new(w, None);
        let active = ActiveAdapter::new();
        let lora = LoraLinear::new(base.clone(), active);
        let x = Tensor::ones((2, 3), DType::F32, &dev)?;
        let expected = base.forward(&x)?;
        let got = lora.forward(&x)?;
        assert_eq!(got.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn active_adapter_adds_scaled_delta() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::zeros((4, 3), DType::F32, &dev)?;
        let base = Linear::new(w, None);
        let active = ActiveAdapter::new();
        let mut lora = LoraLinear::new(base, active.clone());
        // rank=2, alpha=4 => scaling = 2. With ones(2,3)/ones(4,2) adapters and
        // an all-ones input of width 3, lora_a(x) = [3, 3], lora_b(.) = [6, 6, 6, 6]
        // scaled by 2 => [12, 12, 12, 12].
        lora.load_adapter(
            "adapter",
            adapter_vb(2, 3, 4, &dev),
            3,
            4,
            &LoraConfig::new(2, 4.0),
        )?;
        let x = Tensor::ones((1, 3), DType::F32, &dev)?;

        // Adapter not active: behaves like the (zeroed) base layer.
        assert_eq!(lora.forward(&x)?.to_vec2::<f32>()?, [[0., 0., 0., 0.]]);

        active.set(Some("adapter".to_string()));
        assert_eq!(lora.forward(&x)?.to_vec2::<f32>()?, [[12., 12., 12., 12.]]);

        active.set(None);
        assert_eq!(lora.forward(&x)?.to_vec2::<f32>()?, [[0., 0., 0., 0.]]);
        Ok(())
    }

    #[test]
    fn shared_active_handle_switches_multiple_layers() -> Result<()> {
        let dev = Device::Cpu;
        let active = ActiveAdapter::new();
        let mut a = LoraLinear::new(
            Linear::new(Tensor::zeros((2, 2), DType::F32, &dev)?, None),
            active.clone(),
        );
        let mut b = LoraLinear::new(
            Linear::new(Tensor::zeros((2, 2), DType::F32, &dev)?, None),
            active.clone(),
        );
        a.load_adapter(
            "x",
            adapter_vb(1, 2, 2, &dev),
            2,
            2,
            &LoraConfig::new(1, 1.0),
        )?;
        b.load_adapter(
            "x",
            adapter_vb(1, 2, 2, &dev),
            2,
            2,
            &LoraConfig::new(1, 1.0),
        )?;

        assert!(!a.has_adapter("missing"));
        assert!(a.has_adapter("x") && b.has_adapter("x"));

        let x = Tensor::ones((1, 2), DType::F32, &dev)?;
        let off_a = a.forward(&x)?.to_vec2::<f32>()?;
        let off_b = b.forward(&x)?.to_vec2::<f32>()?;

        active.set(Some("x".to_string()));
        let on_a = a.forward(&x)?.to_vec2::<f32>()?;
        let on_b = b.forward(&x)?.to_vec2::<f32>()?;

        assert_ne!(on_a, off_a);
        assert_ne!(on_b, off_b);
        Ok(())
    }
}
