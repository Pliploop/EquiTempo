from config.template import Config


class EvaluationConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.checkpoint_path = "checkpoints"
        self.device = "cuda"
        # this can be overriden by command line args
        self.dataset_name = "acm_mirum"
