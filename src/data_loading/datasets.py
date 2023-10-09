import os
import random

import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
import soxr
import torchaudio.functional as F
import torchaudio.transforms as T
from config.dataset import MTATConfig
from config.preprocessing import PreprocessingConfig
from torch.utils.data import DataLoader, Dataset
import librosa
from src.data_loading.preprocessing import *


class MTATDataset(Dataset):
    def __init__(self, annotations_file = MTATConfig().annotations_path, audio_dir = MTATConfig().dir_path, train = True):
        self.annotations = pd.read_csv(annotations_file, sep='\t')
        self.annotations = self.annotations[~self.annotations.mp3_path.isna()]
        self.audio_dir = audio_dir
        self.train = train
        self.config = MTATConfig()
        self.preprocessing_config = PreprocessingConfig()
        self.melgram = T.MelSpectrogram(sample_rate=44100, f_min=30, f_max=17000, n_mels=81, n_fft=2048, win_length=2048, hop_length=441, power=1)
        ## Move melgram params to config


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):


        audio_path = f"{self.audio_dir}/{self.annotations.mp3_path.iloc[idx]}"

        len_audio_n = self.preprocessing_config.len_audio_n
        len_audio_n_dataset = self.preprocessing_config.len_audio_n_dataset

        info = sf.info(audio_path)
        samplerate = info.samplerate
        duration = info.duration
        length = int(samplerate*duration)

        audio, sample_rate = sf.read(audio_path, frames=len_audio_n_dataset, start=np.random.randint(0, length-len_audio_n_dataset), stop=None, dtype='float32', always_2d=True)
        audio = audio[:,0]

        # audio, sample_rate = torchaudio.load(audio_path, format="mp3")
        if sample_rate != self.preprocessing_config.sr:
            audio = soxr.resample(audio, samplerate, self.preprocessing_config.sr)
            audio = np.expand_dims(audio, 0)
            # F.resample(audio,orig_freq=sample_rate,new_freq=self.preprocessing_config.sr)


        
        

        ## Random sampling of 13.6 seconds of audio

        # if audio.shape[1] > len_audio_n:
        #     start = random.randint(0,audio.shape[1] - len_audio_n)
        #     audio = audio[0,start:start+len_audio_n]

        ## to numpy for sox
        # audio = audio.numpy()
        ## kinda tedious, but torch only stretches specgrams
        ## maybe specgram -> stretch -> specgram?

    

        ## stretching
        rp_range = self.preprocessing_config.rp_range
        rp_1 = np.random.uniform(rp_range[0],rp_range[1])
        rp_2 = np.random.uniform(rp_range[0],rp_range[1])


        audio_1 = librosa.effects.time_stretch(audio,rate=rp_1)
        audio_2 = librosa.effects.time_stretch(audio,rate=rp_2)


        # print(audio_1.shape)
        # print(audio_2.shape)

        ## cropping or padding
        audio_1 = pad_or_truncate(torch.from_numpy(audio_1),len_audio_n)
        audio_2 = pad_or_truncate(torch.from_numpy(audio_2),len_audio_n)

        # print(audio_1.shape)
        # print(audio_2.shape)

        return 0

        ## Augmentations


        # return {
        #     "audio_1" : torch.Tensor(librosa.amplitude_to_db(self.melgram(audio_1))),
        #     "audio_2" : torch.Tensor(librosa.amplitude_to_db(self.melgram(audio_2))),
        #     "rp_1" : rp_1,
        #     "rp_2" : rp_2}

    def create_dataloader(self):
        return DataLoader(dataset=self,batch_size=self.config.batch_size, shuffle=True)

