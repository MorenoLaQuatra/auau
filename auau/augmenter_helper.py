
from auau.augmenter import Augmenter
import numpy as np
import collections
import random
import logging

class AugmenterHelper:
    '''This is the docstring for this class'''

    def __init__(self, noise_folder, device=None):
        super(AugmenterHelper, self).__init__()
        self.augmenter = Augmenter(device=device)
        self.noise_folder = noise_folder

    '''
    Robust Speech Recognition challenge
    0.15 -- Low pass 1000-1500 - apply
    0.15 -- High pass 500-1000 - apply
    0.45 -- Noise (random 8K)  - apply
    0.1 -- noise + LP
    0.1 -- noise + HP
    0.05 -- pitch shifting [-2;+2] random


    How to remove hardcoded?
    '''

    def augment(self, 
        list_dicts: list=[],
        signal_key: str="array",
        sr_key: str="sr"):
        
        list_augmentations = ["LP", "HP", "noise", "noise+HP", "noise+LP", "pitch"]
        list_probabilities = [0.15, 0.15, 0.45, 0.1, 0.1, 0.05]
        
        for i, d in enumerate(list_dicts):
            signal = d[signal_key]
            sr = d[sr_key]
            c = np.random.choice(list_augmentations, p=list_probabilities)

            if c == "LP":
                # range 1000-1500
                augmented_signal = self.augmenter.low_pass_filter(signal, sr, cutoff_freq=random.randint(1000,1200))
            elif c == "HP":
                # range 500-1000
                augmented_signal = self.augmenter.high_pass_filter(signal, sr, cutoff_freq=random.randint(800,1000))
            elif c == "noise":
                augmented_signal = self.augmenter.add_random_noise(signal, sr, noise_folder=self.noise_folder, max_noise_factor=0.1)
            elif c == "noise+LP":
                augmented_signal = self.augmenter.add_random_noise(signal, sr, noise_folder=self.noise_folder, max_noise_factor=0.1)
                augmented_signal = self.augmenter.low_pass_filter(augmented_signal, sr, cutoff_freq=random.randint(1000,1200))
            elif c == "noise+HP":
                augmented_signal = self.augmenter.add_random_noise(signal, sr, noise_folder=self.noise_folder, max_noise_factor=0.1)
                augmented_signal = self.augmenter.high_pass_filter(augmented_signal, sr, cutoff_freq=random.randint(800,1000))
            elif c == "pitch":
                augmented_signal = self.augmenter.pitch_scale(signal, sr, num_semitones=random.randint(-2,2))
            else:
                logging.ERROR("Augmentation not found, return original signal")

            list_dicts[i][signal_key] = augmented_signal
            list_dicts[i]["augmentation_type"] = c

        return list_dicts



