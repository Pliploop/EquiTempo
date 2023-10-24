import torch
import random
from torchaudio.transforms import Vol
import numpy as np
from julius.filters import highpass_filter, lowpass_filter


class RandomApply(torch.nn.Module):
    """Apply randomly a list of transformations with a given probability.

    .. note::
        In order to script the transformation, please use ``torch.nn.ModuleList`` as input instead of list/tuple of
        transforms as shown below:

        >>> transforms = transforms.RandomApply(torch.nn.ModuleList([
        >>>     transforms.ColorJitter(),
        >>> ]), p=0.3)
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    Args:
        transforms (list or tuple or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
class Compose:
    """Data augmentation module that transforms any given data example with a chain of audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        x = self.transform(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "\t{0}".format(t)
        format_string += "\n)"
        return format_string

    def transform(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ComposeMany(Compose):
    """
    Data augmentation module that transforms any given data example randomly
    resulting in N correlated views of the same example
    """

    def __init__(self, transforms, num_augmented_samples):
        self.transforms = transforms
        self.num_augmented_samples = num_augmented_samples

    def __call__(self, x):
        samples = []
        for _ in range(self.num_augmented_samples):
            samples.append(self.transform(x).unsqueeze(dim=0).clone())
        return torch.cat(samples, dim=0)

class FrequencyFilter(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float,
        freq_high: float,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high

    def cutoff_frequency(self, frequency: float) -> float:
        return frequency / self.sample_rate

    def sample_uniform_frequency(self):
        return random.uniform(self.freq_low, self.freq_high)


class HighPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 200,
        freq_high: float = 1200,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = highpass_filter(audio, cutoff=cutoff)
        return audio


class LowPassFilter(FrequencyFilter):
    def __init__(
        self,
        sample_rate: int,
        freq_low: float = 2200,
        freq_high: float = 4000,
    ):
        super().__init__(sample_rate, freq_low, freq_high)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frequency = self.sample_uniform_frequency()
        cutoff = self.cutoff_frequency(frequency)
        audio = lowpass_filter(audio, cutoff=cutoff)
        return audio



class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio = torch.neg(audio)
        return audio



class Noise(torch.nn.Module):
    def __init__(self, min_snr=0.0001, max_snr=0.01):
        """
        :param min_snr: Minimum signal-to-noise ratio
        :param max_snr: Maximum signal-to-noise ratio
        """
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr

    def forward(self, audio):
        std = torch.std(audio)
        noise_std = random.uniform(self.min_snr * std, self.max_snr * std)

        noise = np.random.normal(0.0, noise_std, size=audio.shape).astype(np.float32)

        return audio + noise



class HighLowPass(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        lowpass_freq_low: float = 2200,
        lowpass_freq_high: float = 4000,
        highpass_freq_low: float = 200,
        highpass_freq_high: float = 1200,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self.high_pass_filter = HighPassFilter(
            sample_rate, highpass_freq_low, highpass_freq_high
        )
        self.low_pass_filter = LowPassFilter(
            sample_rate, lowpass_freq_low, lowpass_freq_high
        )

    def forward(self, audio):
        highlowband = random.randint(0, 1)
        if highlowband == 0:
            audio = self.high_pass_filter(audio)
        else:
            audio = self.low_pass_filter(audio)
        return audio



class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio