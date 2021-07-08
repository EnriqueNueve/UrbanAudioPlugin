
"""
main.py
Example: python3 main.py --SAMPLERATE_HZ 22050 --DURATION_S 10 --TOP_K 3
"""

"""

from waggle.data import AudioFolder, Microphone

def main():
    # can now read test audio data from a folder for testing
    dataset = AudioFolder("audio_test_data")
    for sample in dataset:
        print(sample.data)
        print(sample.samplerate)
    # can now record audio data from a microphone
    microphone = Microphone(samplerate=22050)
    sample = microphone.record(duration=5)
    print(sample.data)


"""


######################
# Import waggle modules
######################

#from waggle import plugin
#from waggle.data import AudioFolder, Microphone
import argparse
import logging
import time

######################
# Import main modules
######################

import librosa
from PIL import Image

import numpy as np
import tensorflow as tf

######################
# Globals
######################

CLASS_LABELS = ['air conditioner','car horn','children playing',\
                'dog barking','drilling', 'engine','gun shot',\
               'jackhammer','siren','street music']

######################
# Utils
#####################

def getAudioModel(model_path='audio_lite_model.tflite'):
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()

    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    return model, input_details, output_details

def getTopK(yh,k=1):
    yh_sorted  = yh[0].argsort()[-k:][::-1].tolist()
    return [CLASS_LABELS[k] for k in yh_sorted], yh[0][yh_sorted[:k]]

def LogMelSpectMesh(y,sr=22050):
    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]

    # Zero-padding for clip(size <= 2205)
    if len(y) <= 2205:
        clip = np.concatenate((y, np.zeros(2205 - len(y) + 1)))

    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*sr/1000))
        hop_length = int(round(hop_sizes[i]*sr/1000))
        spec = librosa.feature.melspectrogram(y,sr=22050,n_fft=2205, win_length=window_length,\
                                             hop_length=hop_length,n_mels=128)
        spec = librosa.power_to_db(spec)
        spec = np.asarray(Image.fromarray(spec).resize((250,128)))

        # Scale between [0,1]
        spec = (spec - np.min(spec))/np.ptp(spec)

        specs.append(spec)

    specs = np.array(specs)
    specs = np.moveaxis(specs, 0, 2)
    return specs

def predictModel(model, input_details, output_details,y,k):
    # Process passed data
    data_spect = LogMelSpectMesh(y)
    data_spect = data_spect*255
    tf_img = data_spect.reshape(1,128,250,3)
    tf_img = tf.convert_to_tensor(tf_img, dtype='float32')

    # Pass through model
    model.set_tensor(input_details[0]['index'], tf_img)
    model.invoke()
    yh = model.get_tensor(output_details[0]['index'])

    # Get top k predictions
    yh_k, yh_conf = getTopK(yh,k)
    return yh_k, yh_conf

######################
# Main
#####################

def main():
    # Get parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--SAMPLERATE_HZ", default=22050, type=int, help="Sample rate of audio in Hz")
    parser.add_argument("--DURATION_S", default=10, type=int, help="Duration of audio clip in seconds")
    parser.add_argument("--TOP_K", default=3, type=int, help="Number of top predictions to store")
    args = parser.parse_args()

    # Declare logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    logging.info("starting plugin. sample rate of audio is {} with duration of {} seconds".format(args.SAMPLERATE_HZ,args.DURATION_S))

    #####################

    # Load model
    model, input_details, output_details = getAudioModel()

    # Get sample with open_data_source
    y, _ = librosa.load("street_music_sample.wav",args.SAMPLERATE_HZ)

    # Make a prediction
    yh_k, yh_conf = predictModel(model, input_details, output_details,y,args.TOP_K)

    for i,k in enumerate(yh_k):
        print("Rank: {} | Class: {} | Score: {}".format(i+1,k, np.round(yh_conf[i],4)))

    #####################

    # Init plugin
    plugin.init()
    microphone = Microphone(samplerate=args.SAMPLERATE_HZ)

    while True:
        sample = microphone.record(duration=args.DURATION_S)
        yh_k, yh_conf = predictModel(model, input_details, output_details,sample,args.TOP_K)

        # Publish to plugin
        for i in range(args.TOP_K):
            plugin.publish("rank."+str(i+1)+".class", yh_k[i])
            plugin.publish("rank."+str(i+1)+".prob", yh_conf[i])


if __name__ == '__main__':
    main()
