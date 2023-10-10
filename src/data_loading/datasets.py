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
from config.full import GlobalConfig
from torch.utils.data import DataLoader, Dataset
import librosa
from src.data_loading.preprocessing import *



class MTATDataset(Dataset):
    def __init__(self, train = True, global_config = GlobalConfig()):
        
        self.config = MTATConfig(dict = global_config.MTAT_config)
        self.preprocessing_config = PreprocessingConfig(dict = global_config.preprocessing_config)
        self.extension = self.config.extension
        self.annotations = pd.read_csv(self.config.annotations_path, sep='\t')
        self.annotations = self.annotations[~self.annotations.mp3_path.isna()]
        self.annotations['file_path'] = self.annotations['mp3_path'].apply(lambda x:x[:-3]+self.config.extension)
        self.audio_dir = self.config.dir_path
        self.run_annotations_check()
        
        self.train = train
        self.melgram = T.MelSpectrogram(sample_rate=44100, f_min=30, f_max=17000, n_mels=81, n_fft=2048, win_length=2048, hop_length=441, power=2)

    def run_annotations_check(self):
        mask = (self.audio_dir+'/'+self.annotations.file_path).apply(lambda x:os.path.exists(x))
        print(f'removing {len(mask) - mask.sum()} files from annotations')
        self.annotations = self.annotations[mask]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):


        audio_path = f"{self.audio_dir}/{self.annotations.file_path.iloc[idx]}"

        len_audio_n = self.preprocessing_config.len_audio_n
        len_audio_n_dataset = self.preprocessing_config.len_audio_n_dataset
        if self.extension == 'mp3' or self.extension == 'wav': ### rewrite better for wav
            info = sf.info(audio_path)
            if self.extension == 'mp3':
                length = info.frames-self.preprocessing_config.pad_mp3
            samplerate = info.samplerate
            audio, sample_rate = sf.read(audio_path, stop=None, dtype='float32', always_2d=True)
            start_crop = np.random.randint(0, length-len_audio_n_dataset)
            audio = audio[start_crop:start_crop+len_audio_n_dataset,0]

        if self.extension == "npy":
            audio = np.load(audio_path)
            length = audio.shape[-1]
            sample_rate = self.preprocessing_config.dataset_sr
            start_crop = np.random.randint(0, length-len_audio_n_dataset)
            audio = audio[start_crop:start_crop+len_audio_n_dataset,0]


        if sample_rate != self.preprocessing_config.sr:
            audio = soxr.resample(audio, samplerate, self.preprocessing_config.sr)

        ## stretching
        rp_range = self.preprocessing_config.rp_range
        rp_1 = np.random.uniform(rp_range[0],rp_range[1])
        rp_2 = np.random.uniform(rp_range[0],rp_range[1])


        audio_1 = librosa.effects.time_stretch(audio,rate=rp_1)
        audio_2 = librosa.effects.time_stretch(audio,rate=rp_2)


        ## cropping or padding
        audio_1 = pad_or_truncate(torch.from_numpy(audio_1),len_audio_n)
        audio_2 = pad_or_truncate(torch.from_numpy(audio_2),len_audio_n)

        ## Augmentations


        return {
            "audio_1" : power2db(self.melgram(audio_1)),
            "audio_2" : power2db(self.melgram(audio_2)),
            "rp_1" : rp_1,
            "rp_2" : rp_2}

    def create_dataloader(self):
        return DataLoader(dataset=self,batch_size=self.config.batch_size, shuffle=True)

