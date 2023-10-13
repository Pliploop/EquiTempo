from config.template import Config

class MTATConfig(Config):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.annotations_path = "data/MTAT_annotations.csv"
        self.dir_path = "data/MTAT"
        self.batch_size = 16
        self.extension = 'wav'

        
    