from config.template import Config


class FinetuneConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.output_dim = 300
        self.checkpoint_path = "checkpoints"
        self.lr = 0.0001
        self.epochs = 100
        self.device = "cuda"
        # this can be overriden by command line args
        self.dataset_list = ["gtzan", "giantsteps", "hainsworth"]
