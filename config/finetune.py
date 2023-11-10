from config.template import Config


class FinetuningConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.output_dim = 300
        self.checkpoint_path = "checkpoints"
        self.lr = 0.001
        self.epochs = 2000
        self.device = "cuda"
        self.wandb_log = True
        # this can be overriden by command line args
        self.dataset_name_list = ["gtzan","giantsteps"]
        self.stretch = True
        self.save_path = "checkpoints/fine_tune"
