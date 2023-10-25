from config.template import Config


class MTATConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.annotations_path = "data/MTAT_annotations.csv"
        self.dir_path = "data/MTAT"
        self.batch_size = 16
        self.extension = "wav"


class FinetuningDatasetConfig(Config):
    def __init__(self, *args, **kwards) -> None:
        super().__init__(*args, **kwards)

        self.audio_dirs = {
            "gtzan": "../gtzan",
            "hainsworth": "../hainsworth",
            "giantsteps": "../giantsteps",
            "acm_mirum": "../acm_mirum",
        }
        self.annotation_dirs = {
            "gtzan": "data/gtzan_tempo.csv",
            "hainsworth": "data/hainsworth_tempo.csv",
            "giantsteps": "data/giantsteps_tempo.csv",
            "acm_mirum": "data/acm_mirum_tempo.csv",
        }

        self.batch_size = 16
        self.extension = "wav"


class EvaluationDatasetConfig(Config):
    def __init__(self, *args, **kwards) -> None:
        super().__init__(*args, **kwards)

        self.audio_dirs = {
            "gtzan": "data/gtzan",
            "hainsworth": "data/hainsworth",
            "giantsteps": "data/giantsteps",
            "acm_mirum": "data/acm_mirum",
        }
        self.annotation_dirs = {
            "gtzan": "data/gtzan_tempo.csv",
            "hainsworth": "data/hainsworth_tempo.csv",
            "giantsteps": "data/giantsteps_tempo.csv",
            "acm_mirum": "data/acm_mirum_tempo.csv",
        }

        self.batch_size = 16
        self.extension = "wav"
