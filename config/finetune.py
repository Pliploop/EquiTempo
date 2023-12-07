from config.template import Config


class FinetuningConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        # super().__init__(*args, **kwargs)

        self.output_dim = 300
        self.checkpoint_path = "checkpoints"
        self.lr = 0.003
        self.epochs = 100
        self.device = "cuda"
        self.wandb_log = True
        # this can be overriden by command line args
        self.dataset_name_list = ["gtzan","giantsteps"]
        self.stretch = True
        self.save_path = "checkpoints/fine_tune"
        self.finetuned_from = ''
        self.save_every = 10

        # if called above, the dictionary that is initialized by using the 'dict' argument (see config/template.py) is overwritten
        super().__init__(*args, **kwargs)
