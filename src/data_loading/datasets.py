import os
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from config.dataset import EvaluationDatasetConfig, FinetuningDatasetConfig, MTATConfig
from config.full import GlobalConfig
from config.preprocessing import PreprocessingConfig
from src.data_loading.augmentations import *
from src.data_loading.preprocessing import logcomp, pad_or_truncate, power2db
from src.utils.stretch import stretch


class MTATDataset(Dataset):
    def __init__(self, train=True, global_config=GlobalConfig()):
        self.config = MTATConfig(dict=global_config.MTAT_config)
        self.preprocessing_config = PreprocessingConfig(
            dict=global_config.preprocessing_config
        )
        self.extension = self.config.extension
        self.annotations = pd.read_csv(self.config.annotations_path, sep="\t")
        self.annotations = self.annotations[~self.annotations.mp3_path.isna()]
        self.annotations["file_path"] = self.annotations["mp3_path"].apply(
            lambda x: x[:-3] + self.config.extension
        )
        self.audio_dir = self.config.dir_path
        self.run_annotations_check()

        self.train = train

        self.augment = self.config.augment
        self.melgram = T.MelSpectrogram(
            sample_rate=44100,
            f_min=30,
            f_max=17000,
            n_mels=81,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            power=1,
        )
        self.sr = self.preprocessing_config.sr

        ## Move to config?
        self.audio_transforms = Compose(
            [
                RandomApply([PolarityInversion()], p=self.config.polarity_aug_chance),
                RandomApply(
                    [Noise(min_snr=0.001, max_snr=0.005)],
                    p=self.config.gaussian_noise_aug_chance,
                ),
                RandomApply([Gain()], p=self.config.gain_aug_chance),
                RandomApply(
                    [HighLowPass(sample_rate=self.sr)], p=self.config.filter_aug_chance
                ),
            ]
        )

        self.spectrogram_transforms = Compose(
            [
                RandomApply(
                    [T.TimeMasking(time_mask_param=80)],
                    p=self.config.specaugment_aug_chance,
                ),
                RandomApply(
                    [T.FrequencyMasking(freq_mask_param=80)],
                    p=self.config.specaugment_aug_chance,
                ),
            ]
        )

    def run_annotations_check(self):
        mask = (self.audio_dir + "/" + self.annotations.file_path).apply(
            lambda x: os.path.exists(x)
        )
        print(f"removing {len(mask) - mask.sum()} files from annotations")
        self.annotations = self.annotations[mask]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = f"{self.audio_dir}/{self.annotations.file_path.iloc[idx]}"

        len_audio_n = self.preprocessing_config.len_audio_n
        len_audio_n_dataset = self.preprocessing_config.len_audio_n_dataset
        if (
            self.extension == "mp3" or self.extension == "wav"
        ):  ### rewrite better for wav
            info = sf.info(audio_path)
            if self.extension == "mp3":
                length = info.frames - self.preprocessing_config.pad_mp3
            else:
                length = info.frames
            sample_rate = info.samplerate
            audio, sample_rate = sf.read(
                audio_path, stop=None, dtype="float32", always_2d=True
            )
            start_crop = np.random.randint(0, length - len_audio_n_dataset)

            audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if self.extension == "npy":
            audio = np.load(audio_path)
            length = audio.shape[-1]
            sample_rate = self.preprocessing_config.dataset_sr
            start_crop = np.random.randint(0, length - len_audio_n_dataset)
            audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if sample_rate != self.preprocessing_config.sr:
            audio = soxr.resample(audio, sample_rate, self.preprocessing_config.sr)

        ## stretching
        rp_range = self.preprocessing_config.rp_range

        rp_1 = np.random.uniform(rp_range[0], rp_range[1])
        rp_2 = np.random.uniform(rp_range[0], rp_range[1])

        audio_1 = librosa.effects.time_stretch(audio, rate=rp_1)
        audio_2 = librosa.effects.time_stretch(audio, rate=rp_2)

        ## cropping or padding
        audio_1 = pad_or_truncate(torch.from_numpy(audio_1), len_audio_n)
        audio_2 = pad_or_truncate(torch.from_numpy(audio_2), len_audio_n)

        ## Augmentations
        if self.train and self.augment:
            audio_1 = self.audio_transforms.transform(audio_1)
            audio_2 = self.audio_transforms.transform(audio_2)

        mel1 = self.melgram(audio_1)
        mel2 = self.melgram(audio_2)

        if self.train and self.augment:
            mel1 = self.spectrogram_transforms.transform(mel1)
            mel2 = self.spectrogram_transforms.transform(mel2)

        mel1 = logcomp(mel1)
        mel2 = logcomp(mel2)

        return {"audio_1": mel1, "audio_2": mel2, "rp_1": rp_1, "rp_2": rp_2}

    def create_dataloader(self, split=0.1):
        train_dataset, val_dataset = torch.utils.data.random_split(
            self, [1 - split, split]
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        return train_dataloader, val_dataloader


class FinetuningDataset(Dataset):
    def __init__(
        self,
        dataset_name_list: list,
        stretch: bool = True,
        global_config=GlobalConfig(),
        debug=False,
    ):
        self.config = FinetuningDatasetConfig(
            dict=global_config.finetuning_dataset_config
        )
        self.preprocessing_config = PreprocessingConfig(
            dict=global_config.preprocessing_config
        )
        self.debug = debug
        self.stretch = stretch
        self.extension = self.config.extension
        self.audio_dirs = self.config.audio_dirs
        self.annotations = {
            dataset_name: pd.read_csv(
                self.config.annotation_dirs[dataset_name], sep=","
            )
            for dataset_name in dataset_name_list
        }
        for df in self.annotations.values():
            df["tempo"] = df["tempo"].apply(float)

        # create an audio_path column depending on dataset structure
        for dataset_name in dataset_name_list:
            if dataset_name == "gtzan":
                self.annotations[dataset_name]["audio_path"] = self.annotations[
                    dataset_name
                ]["name"].apply(
                    lambda x: os.path.join(
                        self.audio_dirs["gtzan"],
                        x.split(".")[0],
                        x + "." + self.config.extension,
                    )
                )
            elif dataset_name == "giantsteps":
                self.annotations[dataset_name]["audio_path"] = self.annotations[
                    dataset_name
                ]["name"].apply(
                    lambda x: os.path.join(
                        self.audio_dirs["giantsteps"],
                        str(x) + ".LOFI." + self.config.extension,
                    )
                )
            elif dataset_name == "hainsworth":
                self.annotations[dataset_name]["audio_path"] = self.annotations[
                    dataset_name
                ]["name"].apply(
                    lambda x: os.path.join(
                        self.audio_dirs["hainsworth"],
                        x.split(".")[0] + "." + self.config.extension,
                    )
                )

        # concatente to one dataframe
        self.annotations = pd.concat(
            [self.annotations[dataset_name] for dataset_name in dataset_name_list]
        )

        if stretch:
            # drop all tracks whose tempo times min(rf_range) is greater than 300 bpm
            rf_range = self.preprocessing_config.rf_range
            self.annotations = self.annotations[
                self.annotations["tempo"] * rf_range[0] < 300
            ]
        else:
            self.annotations = self.annotations[self.annotations["tempo"] < 300]

        # print number of tracks
        print(f"Number of tracks for finetuning: {len(self.annotations)}")

        self.melgram = T.MelSpectrogram(
            sample_rate=44100,
            f_min=30,
            f_max=17000,
            n_mels=81,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            power=1,
        )
        if self.config.sanity_check_n:
            self.annotations = self.annotations.head(self.config.sanity_check_n)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = self.annotations.iloc[idx]["audio_path"]

        len_audio_n = self.preprocessing_config.len_audio_n
        if self.extension == "mp3" or self.extension == "wav":
            info = sf.info(audio_path)
            if self.extension == "mp3":
                length = info.frames - self.preprocessing_config.pad_mp3
            else:
                length = info.frames
            sample_rate = info.samplerate
            audio, sample_rate = sf.read(
                audio_path, stop=None, dtype="float32", always_2d=True
            )

            len_audio_n_dataset = int(
                self.preprocessing_config.len_audio_s * sample_rate
            )
            # in Hainsworth you get a shorter audio file than the required 13.6 seconds. In that case, sample another random valid audio
            try:
                start_crop = np.random.randint(0, length - len_audio_n_dataset)
            except Exception as e:
                print(e)
                return self.__getitem__(np.random.randint(0, len(self.annotations) - 1))
            audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if self.extension == "npy":  ### Nothing here for sample rates of other datasets
            audio = np.load(audio_path)
            length = audio.shape[-1]
            sample_rate = self.preprocessing_config.dataset_sr
            start_crop = np.random.randint(0, length - len_audio_n_dataset)
            audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if sample_rate != self.preprocessing_config.sr:
            if self.debug:
                print(
                    f"resampling from {sample_rate} to {self.preprocessing_config.sr}"
                )
            audio = soxr.resample(audio, sample_rate, self.preprocessing_config.sr)

        # time stretching
        if self.stretch:
            rf_range = self.preprocessing_config.rf_range
            rf = np.random.uniform(rf_range[0], rf_range[1])
            # make sure it doesn't exceed 300 bpm
            while rf > 299 / self.annotations.iloc[idx]["tempo"]:
                rf = np.random.uniform(rf_range[0], rf_range[1])
            if self.debug:
                print(f"stretching by factor {rf}")
            # audio = librosa.effects.time_stretch(audio, rate=rf)

            tfm = sox.Transformer()
            tfm.set_globals(verbosity=0)
            if abs(rf - 1.0) <= 0.1:
                tfm.stretch(rf)
            else:
                tfm.tempo(rf, quick=True)
            audio = tfm.build_array(
                input_array=audio, sample_rate_in=self.preprocessing_config.sr
            )

        else:
            rf = 1

        # cropping or padding
        audio = pad_or_truncate(torch.from_numpy(np.copy(audio)), len_audio_n)

        print(audio.shape)
        audio = logcomp(self.melgram(audio))

        return {
            "audio": audio,
            "tempo": self.annotations.iloc[idx]["tempo"],
            "rf": rf,
        }

    def create_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


class EvaluationDataset(Dataset):
    def __init__(
        self, dataset_name: str, stretch: bool = False, global_config=GlobalConfig()
    ):
        self.config = EvaluationDatasetConfig(
            dict=global_config.evaluation_dataset_config
        )
        self.preprocessing_config = PreprocessingConfig(
            dict=global_config.preprocessing_config
        )
        self.stretch = stretch
        self.extension = self.config.extension
        self.audio_dirs = self.config.audio_dirs
        self.annotations = pd.read_csv(
            self.config.annotation_dirs[dataset_name], sep=","
        )

        self.annotations["tempo"] = self.annotations["tempo"].apply(float)

        # create an audio_path column depending on dataset structure
        if dataset_name == "gtzan":
            self.annotations["audio_path"] = self.annotations["name"].apply(
                lambda x: os.path.join(
                    self.audio_dirs["gtzan"],
                    x.split(".")[0],
                    x + "." + self.config.extension,
                )
            )
        elif dataset_name == "giantsteps":
            self.annotations["audio_path"] = self.annotations["name"].apply(
                lambda x: os.path.join(
                    self.audio_dirs["giantsteps"],
                    str(x) + ".LOFI." + self.config.extension,
                )
            )
        elif dataset_name == "hainsworth":
            self.annotations["audio_path"] = self.annotations["name"].apply(
                lambda x: os.path.join(
                    self.audio_dirs["hainsworth"],
                    x.split(".")[0] + "." + self.config.extension,
                )
            )

        if stretch:
            # drop all tracks whose tempo times min(rf_range) is greater than 300 bpm
            rf_range = self.preprocessing_config.rf_range
            self.annotations = self.annotations[
                self.annotations["tempo"] * rf_range[0] < 300
            ]
        else:
            self.annotations = self.annotations[self.annotations["tempo"] < 300]

        self.melgram = T.MelSpectrogram(
            sample_rate=44100,
            f_min=30,
            f_max=17000,
            n_mels=81,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            power=1,
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_path = self.annotations.iloc[idx]["audio_path"]

        len_audio_n = self.preprocessing_config.len_audio_n
        len_audio_n_dataset = self.preprocessing_config.len_audio_n_dataset
        if self.extension == "mp3" or self.extension == "wav":
            info = sf.info(audio_path)
            if self.extension == "mp3":
                length = info.frames - self.preprocessing_config.pad_mp3
            else:
                length = info.frames
            samplerate = info.samplerate
            audio, sample_rate = sf.read(
                audio_path, stop=None, dtype="float32", always_2d=True
            )
            # start_crop = np.random.randint(0, length - len_audio_n_dataset)
            # audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if self.extension == "npy":
            audio = np.load(audio_path)
            length = audio.shape[-1]
            sample_rate = self.preprocessing_config.dataset_sr
            # start_crop = np.random.randint(0, length - len_audio_n_dataset)
            # audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if sample_rate != self.preprocessing_config.sr:
            audio = soxr.resample(audio, samplerate, self.preprocessing_config.sr)

        audio = torch.from_numpy(audio)
        try:
            audio = torch.split(audio, len_audio_n, dim=0)[:-1]
        except Exception as e:
            print(e)
            return self[idx + 1]

        melgram = [logcomp(self.melgram(x.squeeze())) for x in audio]
        if len(melgram) == 0:
            return self[idx + 1]
        melgram = torch.stack(melgram, dim=0)
        # time stretching
        # if self.stretch:
        #     rf_range = self.preprocessing_config.rf_range
        #     rf = np.random.uniform(rf_range[0], rf_range[1])
        #     audio = librosa.effects.time_stretch(audio, rate=rf)
        # else:
        #     rf = 1.0

        # cropping or padding
        # audio = pad_or_truncate(torch.from_numpy(audio), len_audio_n)

        tempo = torch.Tensor([self.annotations.iloc[idx]["tempo"]]).squeeze()
        tempo = tempo.repeat(melgram.shape[0])

        return {
            "audio": melgram,
            "tempo": tempo,
            "rf": torch.ones_like(tempo),
        }

    def create_dataloader(self):
        return DataLoader(dataset=self, batch_size=1, shuffle=False)


class MetaDataset(Dataset):
    """GTZAN meta dataset, split based on genre.

    Meta task design:
    For each metatask, we want to fill all tempo classes. Not all of them will originally be filled,
    and some might have more than one track. If a tempo class is empty, we will randomly choose 1
    track out of the 4 closest ones in tempo, and stretch it to the target tempo.  When more than 1
    track per class exists, we'll get a random one. We'll keep 300  items per dataset (that change
    per epoch since there's a random crop and random closest song), corresponding to the 300 tempo
    classes.
    """

    def __init__(self, genre, n_tracks=300, global_config=GlobalConfig()):
        self.n_tracks = n_tracks
        self.config = EvaluationDatasetConfig(
            dict=global_config.evaluation_dataset_config
        )
        self.preprocessing_config = PreprocessingConfig(
            dict=global_config.preprocessing_config
        )
        self.extension = self.config.extension
        self.audio_dirs = self.config.audio_dirs
        self.annotations = pd.read_csv(self.config.annotation_dirs["gtzan"], sep=",")

        self.annotations["tempo"] = self.annotations["tempo"].apply(float)

        # create an audio_path column
        self.annotations["audio_path"] = self.annotations["name"].apply(
            lambda x: os.path.join(
                self.audio_dirs["gtzan"],
                x.split(".")[0],
                x + "." + self.config.extension,
            )
        )

        # keep only tracks of the specified genre. the genre is included in the audiopath
        self.annotations = self.annotations[
            self.annotations["audio_path"].str.contains(genre)
        ]

        # keep all tracks whose tempo is under 300
        self.annotations = self.annotations[self.annotations["tempo"] < 300]

        # sort annotations by tempo
        self.annotations = self.annotations.sort_values(by=["tempo"])

        self.melgram = T.MelSpectrogram(
            sample_rate=44100,
            f_min=30,
            f_max=17000,
            n_mels=81,
            n_fft=2048,
            win_length=2048,
            hop_length=441,
            power=1,
        )

    def __len__(self):
        return self.n_tracks

    def __getitem__(self, idx):  # the index refers to the tempo class
        # ONLY WORKS FOR BATCH_SIZE = 1
        len_audio_n = self.preprocessing_config.len_audio_n

        # if tempo < 24, use an empty signal
        if idx < 24:
            audio = np.array([0.0])
            # cropping or padding
            audio = pad_or_truncate(torch.from_numpy(audio), len_audio_n)

            melgram = self.melgram(audio)
            melgram = logcomp(melgram)
            if len(melgram) == 0:
                return self[idx]

            return {
                "audio": melgram,
                "tempo": idx,
                "rf": 1,
            }

        # get track whose int(round(tempo)) is equal to idx
        try:
            audio_path = (
                self.annotations[round(self.annotations["tempo"]) == idx]["audio_path"]
                .sample(
                    n=1
                )  # get a random one if multiple, not the first one necessarily
                .iloc[0]
            )
            tempo = idx
        except Exception:
            # there's not track with this tempo, so select a random track out of the 3
            # with the closest tempo to idx
            audio_path = (
                self.annotations.iloc[
                    (self.annotations["tempo"] - idx).abs().argsort()[:3]
                ]["audio_path"]
                .sample(n=1)
                .iloc[0]
            )

            # get tempo
            track_idx = self.annotations[
                self.annotations["audio_path"] == audio_path
            ].index[0]
            tempo = self.annotations.loc[track_idx, "tempo"]

        # read audio
        if self.extension == "mp3" or self.extension == "wav":
            info = sf.info(audio_path)
            if self.extension == "mp3":
                length = info.frames - self.preprocessing_config.pad_mp3
            else:
                length = info.frames
            samplerate = info.samplerate
            audio, sample_rate = sf.read(
                audio_path, stop=None, dtype="float32", always_2d=True
            )
            # start_crop = np.random.randint(0, length - len_audio_n_dataset)
            # audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if self.extension == "npy":
            audio = np.load(audio_path)
            length = audio.shape[-1]
            sample_rate = self.preprocessing_config.dataset_sr
            # start_crop = np.random.randint(0, length - len_audio_n_dataset)
            # audio = audio[start_crop : start_crop + len_audio_n_dataset, 0]

        if sample_rate != self.preprocessing_config.sr:
            audio = soxr.resample(audio, samplerate, self.preprocessing_config.sr)

        audio = audio.squeeze()

        # stretch if not exactly the target tempo
        rf = 1
        if int(round(tempo)) != idx:
            rf = idx / tempo
            audio = librosa.effects.time_stretch(audio, rate=rf)

        # cropping or padding
        audio = pad_or_truncate(torch.from_numpy(audio), len_audio_n)

        melgram = self.melgram(audio)
        if len(melgram) == 0:
            return self[idx]

        melgram = logcomp(melgram)

        return {
            "audio": melgram,
            "tempo": idx,
            "rf": rf,
        }

    def create_dataloader(self):
        return DataLoader(dataset=self, batch_size=25, shuffle=True)
