#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Activation, Module};

/// build a stack of alternating `M` and `A` layers
pub struct StackLayers<M>
where
    M: Module,
{
    module_layers: Vec<M>,
    activation_layers: Vec<Option<Activation>>,
}

impl<M> Module for StackLayers<M>
where
    M: Module,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (module, activation) in self.module_layers.iter().zip(self.activation_layers.iter()) {
            x = module.forward(&x)?;
            if let Some(activation) = activation {
                x = activation.forward(&x)?;
            }
        }
        Ok(x)
    }
}

impl<M> StackLayers<M>
where
    M: Module,
{
    pub fn new() -> Self {
        Self {
            module_layers: Vec::new(),
            activation_layers: Vec::new(),
        }
    }

    /// Appends a layer after all the current layers.
    pub fn push_with_act(&mut self, layer: M, activation: Activation) {
        self.module_layers.push(layer);
        self.activation_layers.push(Some(activation));
    }

    pub fn push(&mut self, layer: M) {
        self.module_layers.push(layer);
        self.activation_layers.push(None);
    }
}

impl<M> Default for StackLayers<M>
where
    M: Module,
{
    fn default() -> Self {
        Self::new()
    }
}
