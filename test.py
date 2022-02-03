from auau.augmenter import Augmenter
from auau.loader import Loader
import soundfile as sf
import librosa

au = Augmenter()
loader = Loader()

signal, sr = loader.load_file("resources/test_audio.wav")

# White noise
noise_factor = 0.25
augmented_signal = au.add_white_noise(signal, noise_factor=0.25)
loader.save_file(f"outputs/test_audio_noise-{noise_factor}.wav", augmented_signal, sr)

# Time Strech
time_stretch_rate = 1.1
augmented_signal = au.time_stretch(signal, time_stretch_rate=time_stretch_rate)
loader.save_file(f"outputs/test_audio_timestretch-{time_stretch_rate}.wav", augmented_signal, sr)

# Pitch Scale
num_semitones = 5
augmented_signal = au.pitch_scale(signal, sr=sr, num_semitones=num_semitones)
loader.save_file(f"outputs/test_audio_pitch-{num_semitones}.wav", augmented_signal, sr)

# User-provided gain
augmented_signal = au.gain(signal, 3.14)
loader.save_file("outputs/test_audio_gain.wav", augmented_signal, sr)

# Random Gain
augmented_signal = au.random_gain(signal, min_gain=1, max_gain=1.1)
loader.save_file(f"outputs/test_audio_random_gain.wav", augmented_signal, sr)

# Polarity
augmented_signal = au.invert_polarity(signal)
loader.save_file(f"outputs/test_audio_invert_polarity.wav", augmented_signal, sr)

# External noise
noise_signal, noise_sr = librosa.load("noise/7383-3-0-0.wav")
au.resample(noise_signal, source_sr = noise_sr, target_sr = sr)
augmented_signal = au.add_noise(signal, noise_signal, noise_factor=0.25)
loader.save_file(f"outputs/test_noise.wav", augmented_signal, sr)

# Multiple external noise
list_noise_signals = []
list_noise_factors = []

noise_signal, noise_sr = librosa.load("noise/7383-3-0-1.wav")
list_noise_signals.append(librosa.resample(noise_signal, noise_sr, sr))
list_noise_factors.append(0.1)

noise_signal, noise_sr = librosa.load("noise/14113-4-0-0.wav")
list_noise_signals.append(librosa.resample(noise_signal, noise_sr, sr))
list_noise_factors.append(0.15)

augmented_signal = au.add_multiple_noise(signal, list_noise_signals=list_noise_signals, list_noise_factors=list_noise_factors)
loader.save_file(f"outputs/test_multiple_noise.wav", augmented_signal, sr)


# Random noise from folder
augmented_signal = au.add_random_noise(signal, sr, noise_folder="./noise/", min_noise_factor=0.01, max_noise_factor=0.05)
loader.save_file(f"outputs/test_random_noise.wav", augmented_signal, sr)

# Low pass filter
augmented_signal = au.low_pass_filter(signal, sr)
loader.save_file(f"outputs/test_audio_lowpass.wav", augmented_signal, sr)

# High pass filter
augmented_signal = au.high_pass_filter(signal, sr)
loader.save_file(f"outputs/test_audio_highpass.wav", augmented_signal, sr)

# Flanger
augmented_signal = au.flanger(signal, sr)
loader.save_file(f"outputs/test_audio_flanger.wav", augmented_signal, sr)

# Phase
augmented_signal = au.phaser(signal, sr)
loader.save_file(f"outputs/test_audio_phaser.wav", augmented_signal, sr)

