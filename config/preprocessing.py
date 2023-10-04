from config.template import Config

class PreprocessingConfig(Config):

    def __init__(self) -> None:
        super().__init__()
        self.sr = 44100
        self.len_audio_s = 13.6
        self.len_audio_n = int(self.sr * self.len_audio_s)
        # self.len_audio_n = 600000
        self.rp = 0.1
        self.rp_range = [1-self.rp,1+self.rp]
