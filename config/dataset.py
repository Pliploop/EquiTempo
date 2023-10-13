from config.template import Config

class MTATConfig(Config):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.annotations_path = "/import/c4dm-datasets/MagnaTagATune/annotations_final.csv"
        self.dir_path = "/import/research_c4dm/JulienMarcoChrisRMRI/MTAT_wav"
        self.batch_size = 16
        self.extension = 'wav'

        
    