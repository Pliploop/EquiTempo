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
from src.data_loading.augmentations import *
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
        self.augment = self.config.augment
        self.melgram = T.MelSpectrogram(sample_rate=44100, f_min=30, f_max=17000, n_mels=81, n_fft=2048, win_length=2048, hop_length=441, power=2)
        self.sr = self.preprocessing_config.sr

        ## Move to config?
        self.audio_transforms = Compose([
            RandomApply([PolarityInversion()], p=self.config.polarity_aug_chance),
            RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=self.config.gaussian_noise_aug_chance),
            RandomApply([Gain()], p=self.config.gain_aug_chance),
            RandomApply([HighLowPass(sample_rate=self.sr)], p = self.config.filter_aug_chance)
        ])

        self.spectrogram_transforms = Compose([
            RandomApply([T.TimeMasking(time_mask_param=80)],p=self.config.specaugment_aug_chance),
            RandomApply([T.FrequencyMasking(freq_mask_param=80)],p=self.config.specaugment_aug_chance)
        ])

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
            else:
                length = info.frames
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

        if self.train and self.augment:
            audio_1 = self.audio_transforms.transform(audio_1)
            audio_2 = self.audio_transforms.transform(audio_2)

        mel1 = self.melgram(audio_1)
        mel2 = self.melgram(audio_2)

        if self.train and self.augment :
            mel1 = self.spectrogram_transforms.transform(mel1)
            mel2 = self.spectrogram_transforms.transform(mel2)    

        mel1 = power2db(mel1)  
        mel2 = power2db(mel2)  

        return {
            "audio_1" : mel1,
            "audio_2" : mel2,
            "rp_1" : rp_1,
            "rp_2" : rp_2}

    def create_dataloader(self):
        return DataLoader(dataset=self,batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, drop_last=True)
