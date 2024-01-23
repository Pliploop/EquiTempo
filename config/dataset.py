from config.template import Config


class MTATConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.annotations_path = "data/MTAT_annotations.csv"
        self.dir_path = "/import/research_c4dm/JulienMarcoChrisRMRI/MTAT_wav"
        self.batch_size = 1
        self.num_workers = 16
        self.extension = "wav"

        self.augment = True
        self.polarity_aug_chance = 0.2
        self.gain_aug_chance = 0.2
        self.gaussian_noise_aug_chance = 0.2
        self.specaugment_aug_chance = 0.2
        self.reverb_aug_chance = 0.2
        self.filter_aug_chance = 0.3


class FinetuningDatasetConfig(Config):
    def __init__(self, *args, **kwards) -> None:
        super().__init__(*args, **kwards)

        self.audio_dirs = {
            "gtzan": "/import/c4dm-datasets/gtzan_torchaudio/genres",
            "hainsworth": "/import/c4dm-datasets/hainsworth",
            "giantsteps": "/import/research_c4dm/JulienMarcoChrisRMRI/giantsteps-tempo-dataset/audio_wav",
            "acm_mirum": "../acm_mirum",
        }
        self.annotation_dirs = {
            "gtzan": "data/gtzan_tempo.csv",
            "hainsworth": "data/hainsworth_tempo.csv",
            "giantsteps": "data/giantsteps_tempo.csv",
            "acm_mirum": "data/acm_mirum_tempo.csv",
        }

        self.batch_size = 16
        self.num_workers = 8  # 16
        self.extension = "wav"
        self.sanity_check_n = None


class EvaluationDatasetConfig(Config):
    def __init__(self, *args, **kwards) -> None:
        super().__init__(*args, **kwards)

        self.audio_dirs = {
            "gtzan": "/import/c4dm-datasets/gtzan_torchaudio/genres",
            "hainsworth": "/import/c4dm-datasets/hainsworth",
            "giantsteps": "/import/research_c4dm/JulienMarcoChrisRMRI/giantsteps-tempo-dataset/audio_wav",
            "acm_mirum": "../acm_mirum",
        }
        self.annotation_dirs = {
            "gtzan": "data/gtzan_tempo.csv",
            "hainsworth": "data/hainsworth_tempo.csv",
            "giantsteps": "data/giantsteps_tempo.csv",
            "acm_mirum": "data/acm_mirum_tempo.csv",
        }

        self.batch_size = 25
        self.num_workers = 25
        self.extension = "wav"
