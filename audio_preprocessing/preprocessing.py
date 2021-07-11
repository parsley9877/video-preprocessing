import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from python_speech_features import mfcc
from scipy.io import wavfile
import numpy
import librosa


class MFCC(keras.Model):
    def __init__(self, specs: dict, delta: int):
        self.specs = specs
        self.delta = delta

    def process(self, signal: tf.Tensor):
        signal_numpy = signal.numpy()
        mfcc_numpy = mfcc(signal_numpy, **self.specs)
        if self.delta == 0:
            result = mfcc_numpy
            pass
        elif self.delta == 1:
            d_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=1)
            result = np.stack([mfcc_numpy, d_mfcc_numpy], axis=-1)
        elif self.delta == 2:
            d_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=1)
            dd_mfcc_numpy = librosa.feature.delta(mfcc_numpy, order=2)
            result = np.stack([mfcc_numpy, d_mfcc_numpy, dd_mfcc_numpy], axis=-1)

        return tf.constant(result, dtype=tf.float32)

    def call(self, signal):
        signal_numpy = signal.numpy()
        return tf.constant(mfcc(signal_numpy, **self.specs), dtype=tf.float32)



if __name__ == '__main__':
    rate, data = wavfile.read('./audio.wav')
    with open('./MFCC_config.json') as json_obj:
        mfcc_config = json.load(json_obj)
    mfcc_config['winfunc'] = eval(mfcc_config['winfunc'])
    mfcc_obj = MFCC(mfcc_config, delta=2)
    mfcc_out = mfcc_obj.process(tf.constant(data, dtype=tf.float32))
    print(mfcc_out)
    print(mfcc_out.shape)
    print(type(mfcc_out))

