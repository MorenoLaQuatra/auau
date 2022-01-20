import random
import librosa
import numpy as np
import soundfile as sf
import copy
import logging

class Augmenter:
    '''This is the docstring for this class'''

    def __init__(self):
        super(Augmenter, self).__init__()

    def add_white_noise(self, signal, noise_factor=0.25):
        noise_signal = np.random.normal(0, signal.std(), signal.size)
        augmented_signal = signal + noise_signal * noise_factor
        return augmented_signal

    def time_stretch(self, signal, time_stretch_rate=1.5):
        """Time stretching implemented with librosa:
        https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
        """
        return librosa.effects.time_stretch(signal, time_stretch_rate)

    def pitch_scale(self, signal, sampling_rate, num_semitones=2):
        """Pitch scaling implemented with librosa:
        https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
        """
        return librosa.effects.pitch_shift(signal, sampling_rate, num_semitones)

    def random_gain(self, signal, min_gain=1, max_gain=1.2):
        gain_rate = random.uniform(min_gain, max_gain)
        augmented_signal = signal * gain_rate
        return augmented_signal

    def invert_polarity(self, signal):
        return signal * -1

    def add_noise(self, signal, noise, noise_factor=0.25):
        #match the length of the original signal (noise is repeated)
        noise_signal = np.resize(noise, len(signal))
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
            noise_signal = np.resize(noise_signal, len(signal))
            augmented_signal = augmented_signal + noise_signal * list_noise_factors[noise_index]
        return augmented_signal
