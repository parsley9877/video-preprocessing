import json

import tensorflow as tf
from tensorflow import keras
from python_speech_features import mfcc
from scipy.io import wavfile
import numpy

class MFCC(keras.Model):
    def __init__(self, specs: dict):
        self.specs = specs

    def process(self, signal: tf.Tensor):
        signal_numpy = signal.numpy()
        return tf.constant(mfcc(signal_numpy, **self.specs), dtype=tf.float32)

    def call(self, signal):
        signal_numpy = signal.numpy()
        return tf.constant(mfcc(signal_numpy, **self.specs), dtype=tf.float32)



if __name__ == '__main__':
    rate, data = wavfile.read('./audio.wav')
    with open('./MFCC_config.json') as json_obj:
        mfcc_config = json.load(json_obj)
    mfcc_config['winfunc'] = eval(mfcc_config['winfunc'])
    mfcc_obj = MFCC(mfcc_config)
    mfcc_out = mfcc_obj.process(tf.constant(data, dtype=tf.float32))
    print(mfcc_out)
    print(mfcc_out.shape)
    print(type(mfcc_out))

