import librosa
import soundfile

class Loader:
    '''This is the docstring for this class'''

    def __init__(self):
        super(Loader, self).__init__()


    def load_file(self, filename):
        return librosa.load(filename)

    def save_file(self, filename, signal, sampling_rate):
        soundfile.write(filename, signal, sampling_rate)
        return