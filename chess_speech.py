import numpy as np
import time
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from scipy.io.wavfile import write

import pyaudio
from queue import Queue
from threading import Thread
import sys
import time

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


#debe de recibir los modelos ya compilados y cargados para evitar que cada que se llame
#el metodo se tarde en el proceso de carga

silence_threshold = 30 #100
record_act=[]
record1=[]
run = True

# Queue to communiate between the audio callback and main thread
q = Queue()

# Use 1101 for 2sec input audio
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model

chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

# Run the demo for a timeout seconds
#timeout = time.time() + .5*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)



def detect_triggerword_spectrum(x,model):
    """
    Function to predict the location of the trigger word.
    
    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    # the spectogram outputs  and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.8 ): #.5
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.
    
    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

def get_spectrogram(data):
    """
    Function to compute a spectrogram.
    
    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


######################################################################################
#Metodos para detectar la palabra activate
######################################################################################
def record_activate(time_out ,model): 

    # Run the demo for a timeout seconds
    global timeout, silence_threshold, data, record_act, q, run

    timeout = time_out

    record_act=[]

    run=True

    stream = get_audio_input_stream(callback_act)
    stream.start_stream()

    try:
        while run:
            print(".")
            data = q.get()
            spectrum = get_spectrogram(data)
            preds_act = detect_triggerword_spectrum(spectrum,model)
            new_trigger_act = has_new_triggerword(preds_act, chunk_duration, feed_duration)
            if new_trigger_act:
                #sys.stdout.write('1')
                record_act.append("1")
                run=False
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False

    stream.stop_stream()
    stream.close()
    return record_act

def callback_act(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold    
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        #sys.stdout.write('-')
        record_act.append("-")
        return (in_data, pyaudio.paContinue)
    else:
        #sys.stdout.write('.')
        record_act.append(".")
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

######################################################################################
#Metodos para detectar las palabras usadas para las coordenadas
######################################################################################
def record_words(time_out ,models): 

    # Run the demo for a timeout seconds
    global timeout, silence_threshold, data, record_act, q, run
    data = np.zeros(feed_samples, dtype='int16')
    timeout = time_out

    record_act=[]

    run=True

    stream = get_audio_input_stream(callback_words)
    stream.start_stream()

    try:
        while run:
            print(".")
            data = q.get()
            spectrum = get_spectrogram(data)
            preds_act = detect_triggerword_spectrum(spectrum,models[0])
            new_trigger_act = has_new_triggerword(preds_act, chunk_duration, feed_duration)
            if new_trigger_act:
                #sys.stdout.write('1')
                record_act.append("1")
                run=False
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False

    stream.stop_stream()
    stream.close()
    return record_act

def callback_words(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold    
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        #sys.stdout.write('-')
        record_act.append("-")
        return (in_data, pyaudio.paContinue)
    else:
        #sys.stdout.write('.')
        record_act.append(".")
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)