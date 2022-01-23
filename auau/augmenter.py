import random
import librosa
import numpy as np
import soundfile as sf
import copy
import logging
import torch
import math
import os
from auau.loader import Loader
from torch_audiomentations import HighPassFilter, LowPassFilter, Compose
import torchaudio.functional as FAudio

class Augmenter:
    '''This is the docstring for this class'''

    def __init__(self, device=None):
        super(Augmenter, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.loader = Loader()
        self.allowed_extensions = ["wav", "mp3", ]
    
    def _convert_to_numpy(self, signal):
        if torch.is_tensor(signal):
            return signal.numpy()
        else:
            return signal
    
    def _convert_to_tensor(self, signal):
        if not torch.is_tensor(signal):
            return torch.from_numpy(signal)
        else:
            return signal

    def _move_to_cpu(self, signal):
        if torch.is_tensor(signal):
            if signal.device != "cpu":
                return signal.to("cpu")
            else:
                return signal
        else:
            return self._convert_to_tensor(signal)

    def _move_to_device(self, signal):
        return signal.to(self.device)

    def _tensor_on_device(self, signal):
        signal = self._convert_to_tensor(signal)
        signal = self._move_to_device(signal)
        return signal

    def _array_on_cpu(self, signal):
        signal = self._move_to_cpu(signal)      
        signal = self._convert_to_numpy(signal)
        return signal

    def _check_and_convert_mono(self, signal):
        if len(signal.shape) > 1:
            return torch.mean(signal, dim=1)
        else:
            return signal

    def resample(self, signal, source_sr, target_sr):
        signal = self._array_on_cpu(signal)
        signal = self._check_and_convert_mono(signal)
        return librosa.resample(signal, source_sr, target_sr)

    def add_white_noise(self, signal, noise_factor=0.25):
        """ Adding white noise, processing on self.device
        """
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        noise_signal = torch.normal(0, signal.std(), signal.size()).to(self.device)
        augmented_signal = signal + noise_signal * noise_factor
        return augmented_signal

    def time_stretch(self, signal, time_stretch_rate=1.5):
        """Time stretching implemented with librosa, processing on CPU
        """
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        signal = self._array_on_cpu(signal)
        augmented_signal = librosa.effects.time_stretch(signal, time_stretch_rate)
        return self._tensor_on_device(augmented_signal)

    def pitch_scale(self, signal, sampling_rate, num_semitones=2):
        """Pitch scaling implemented with librosa, processing on CPU
        """
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        augmented_signal = FAudio.pitch_shift(signal, sampling_rate, num_semitones)
        return self._tensor_on_device(augmented_signal)

    def random_gain(self, signal, min_gain=1, max_gain=1.2):
        """Random gain, processing on device
        """
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        gain_rate = torch.FloatTensor(signal.shape).uniform_(min_gain, max_gain)
        augmented_signal = signal * gain_rate
        return augmented_signal

    def invert_polarity(self, signal):
        """Invert polarity, processing on device
        """
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        return signal * -1

    def add_noise(self, signal, noise, noise_factor=0.25):
        """Add provided noise, processing on device
        """
       
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        noise  = self._tensor_on_device(noise)
        noise = self._check_and_convert_mono(noise)

        repeats = math.ceil(len(signal) / len(noise))
        noise_signal = copy.deepcopy(noise)
        for n in range(repeats-1):
            noise_signal = torch.cat((noise_signal, noise))
        noise_signal = noise_signal[:len(signal)]
        augmented_signal = signal + noise_signal * noise_factor

        return augmented_signal

    def add_multiple_noise(self, signal, list_noise_signals=None, list_noise_factors=None):
        #match the length of the original signal (noise is repeated)
        if list_noise_signals == None or list_noise_factors==None:
            logging.ERROR("add_multiple_noise: list_noise_signals or list_noise_factors is set to None.")
            return signal
        
        if (len(list_noise_signals) != len(list_noise_factors)):
            logging.ERROR(f"add_multiple_noise: both list_noise_signals and list_noise_factors must have the same length {len(list_noise_signals)}!={len(list_noise_factors)}")
            return signal

        augmented_signal = copy.deepcopy(signal)
        for noise_index, noise_signal in enumerate(list_noise_signals):
            augmented_signal = self.add_noise(augmented_signal, noise_signal, list_noise_factors[noise_index])
        return augmented_signal

    def add_random_noise(self, signal, signal_sr, noise_folder, traverse_folder=True, min_noise_factor=0.05, max_noise_factor=0.15):
        files = librosa.util.find_files(noise_folder, recurse=traverse_folder)
        noise_file = random.choice(files)
        noise_signal, noise_sr = self.loader.load_file(noise_file)
        noise_signal = self.resample(noise_signal, noise_sr, signal_sr)
        noise_signal = self._tensor_on_device(noise_signal)
        noise_signal = self._check_and_convert_mono(noise_signal)
        noise_factor = random.uniform(min_noise_factor, max_noise_factor)
        return self.add_noise(signal, noise_signal, noise_factor)

    '''
    From torchaudio.functional
    '''

    def low_pass_filter(self, signal, sr, cutoff_freq=1000):
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        return FAudio.lowpass_biquad(signal, sr, cutoff_freq=float(cutoff_freq))

    def high_pass_filter(self, signal, sr, cutoff_freq=1000):
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        return FAudio.highpass_biquad(signal, sr, cutoff_freq=float(cutoff_freq))

    def flanger(self, signal, sampling_rate):
        signal = self._tensor_on_device(signal)
        signal = self._check_and_convert_mono(signal)
        signal = torch.unsqueeze(signal, dim=0)
        augmented_signal = FAudio.flanger(signal, sampling_rate)
        return augmented_signal[0, :]

