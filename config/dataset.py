from config.template import Config

class MTATConfig(Config):

    def __init__(self) -> None:
        super().__init__()

        self.annotations_path = "data/MTAT_annotations.csv"
        self.dir_path = "data/MTAT"
        self.batch_size = 16

        
    