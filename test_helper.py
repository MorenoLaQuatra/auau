from auau.augmenter_helper import AugmenterHelper
from auau.loader import Loader

ah = AugmenterHelper(noise_folder="./noise/", device="cpu")
loader = Loader()

example_list = [
    {"array": loader.load_file("resources/common_voice_it_22274222.mp3")[0], "sr": loader.load_file("resources/common_voice_it_22274222.mp3")[1]},
    {"array": loader.load_file("resources/common_voice_it_25565079.mp3")[0], "sr": loader.load_file("resources/common_voice_it_25565079.mp3")[1]},
    {"array": loader.load_file("resources/common_voice_it_26632586.mp3")[0], "sr": loader.load_file("resources/common_voice_it_26632586.mp3")[1]},
    {"array": loader.load_file("resources/common_voice_it_19975804.mp3")[0], "sr": loader.load_file("resources/common_voice_it_19975804.mp3")[1]},
    {"array": loader.load_file("resources/common_voice_it_21886466.mp3")[0], "sr": loader.load_file("resources/common_voice_it_21886466.mp3")[1]},
]

augmented_list = ah.augment(example_list, signal_key="array", sr_key="sr")

for i, d in enumerate(augmented_list):
    augmentation_type = d["augmentation_type"]
    loader.save_file(f"outputs/example_{i}_{augmentation_type}.wav", d["array"], d["sr"])
