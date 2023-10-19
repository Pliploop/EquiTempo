from config.template import Config


class FinetuneConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.checkpoint_path = "checkpoints"
        self.device = "cuda"
        # this can be overriden by command line args
        self.dataset = "acm_mirum"
