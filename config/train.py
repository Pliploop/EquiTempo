from config.template import Config

class TrainConfig(Config):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mixed_precision = True
        self.filters = 16
        self.dilations = [2**k for k in range(10)]
        self.dropout_rate = 0.1
        self.output_dim = 300
        self.checkpoint_path = 'checkpoints'
        self.lr = 0.001
        self.epochs = 1000
        self.device = 'cuda'
        self.display_progress_every = 1
        self.warmup = False
        self.log_wandb = True

        self.exp_name = f'filters_{self.filters}_do_{self.dropout_rate}_od_{self.output_dim}'
        self.save_path = f'{self.checkpoint_path}/{self.exp_name}'


