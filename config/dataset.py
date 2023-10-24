from config.template import Config

class MTATConfig(Config):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.annotations_path = "data/MTAT_annotations.csv"
        self.dir_path = "/import/research_c4dm/JulienMarcoChrisRMRI/MTAT_wav"
        self.batch_size = 16
        self.num_workers = 16
        self.extension = 'wav'


        self.augment = True
        self.polarity_aug_chance = 0.2
        self.gain_aug_chance = 0.2
        self.gaussian_noise_aug_chance = 0.2
        self.specaugment_aug_chance = 0.2
        self.reverb_aug_chance = 0.2
        self.filter_aug_chance = 0.3
        
    