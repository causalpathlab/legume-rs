pub struct TrainConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub num_pretrain_epochs: usize,
    pub device: candle_core::Device,
    pub verbose: bool,
}
