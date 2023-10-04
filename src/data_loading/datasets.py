import os
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from config.dataset import MTATConfig
from config.preprocessing import PreprocessingConfig
from torch.utils.data import DataLoader, Dataset
import librosa
from src.data_loading.preprocessing import *


class MTATDataset(Dataset):
    def __init__(self, annotations_file = MTATConfig().annotations_path, audio_dir = MTATConfig().dir_path, train = True):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.train = train
        self.config = MTATConfig()
        self.preprocessing_config = PreprocessingConfig()


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        audio_path = f"{self.audio_dir}/{self.annotations.mp3_path[idx]}"
        audio, sample_rate = torchaudio.load(audio_path, format="mp3")
        if sample_rate != self.preprocessing_config.sr:
            F.resample(audio,orig_freq=sample_rate,new_freq=self.preprocessing_config.sr)


        
        len_audio_n = self.preprocessing_config.len_audio_n

        ## Random sampling of 13.6 seconds of audio

        if audio.shape[1] > len_audio_n:
            start = random.randint(0,audio.shape[1] - len_audio_n)
            audio = audio[0,start:start+len_audio_n]

        ## to numpy for sox
        audio = audio.numpy()
        ## kinda tedious, but torch only stretches specgrams
        ## maybe specgram -> stretch -> specgram?

        print(audio.shape)
    

        ## stretching
        rp_range = self.preprocessing_config.rp_range
        rp_1 = np.random.uniform(rp_range[0],rp_range[1])
        rp_2 = np.random.uniform(rp_range[0],rp_range[1])

        audio_1 = librosa.effects.time_stretch(audio,rate=rp_1)
        audio_2 = librosa.effects.time_stretch(audio,rate=rp_2)


        ## cropping or padding
        audio_1 = pad_or_truncate(torch.tensor(audio_1),len_audio_n)
        audio_2 = pad_or_truncate(torch.tensor(audio_2),len_audio_n)


        ## Augmentations

        melgram = T.MelSpectrogram(sample_rate=44100, f_min=30, f_max=17000, n_mels=81, n_fft=2048, win_length=2048, hop_length=441, power=1)

        return librosa.amplitude_to_db(melgram(audio_1)),librosa.amplitude_to_db(melgram(audio_2))

    def create_dataloader(self):
        return DataLoader(dataset=self,batch_size=MTATConfig.batch_size, shuffle=True)
