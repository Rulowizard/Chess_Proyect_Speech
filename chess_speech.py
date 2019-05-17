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
import wave
from queue import Queue
from threading import Thread
import sys
import time

import itertools

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


#debe de recibir los modelos ya compilados y cargados para evitar que cada que se llame
#el metodo se tarde en el proceso de carga

silence_threshold = 30 #100
record_act=[]
#Guarda resultado de "Move"
record_m=[]
#Guardan resultados de las letras
a=[]
b=[]
c=[]
d=[]
e=[]
f=[]
g=[]
h=[]
#Guardan resultados de los números
one=[]
two=[]
three=[]
four=[]
five=[]
six=[]
seven=[]
eigth=[]
run = True

# Queue to communiate between the audio callback and main thread
q = Queue()
q2= Queue()

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
            if len(record_act)>10:
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
def grabar_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK_DURATION = 0.5
    CHUNK = int( RATE * CHUNK_DURATION )
    RECORD_SECONDS = 6
    WAVE_OUTPUT_FILENAME = "./recordings/audio.wav"
    
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")
    
    
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    #Archivo con longitud de 6s
    file = ( "./recordings/audio.wav" )
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(file)[:10000]
    segment = padding.overlay(segment)
    segment = segment.set_frame_rate(44100)
    #Genero archivo con longitud de 10s
    segment.export("./recordings/test_mod_audio.wav", format="wav")

    return "Ok"


def record_words(models):
    global record_m
    global a,b,c,d,e,f,g,h
    global one,two,three,four,five,six,seven,eigth

    file = "./recordings/test_mod_audio.wav"
    
    record_m = detect_triggerword(file,models[0])
    record_m = generate_array(record_m,.8)
    
    a = detect_triggerword(file,models[1])
    a = generate_array(a,.8)
    b = detect_triggerword(file,models[1])
    b = generate_array(b,.8)
    c = detect_triggerword(file,models[1])
    c = generate_array(c,.8)
    d = detect_triggerword(file,models[1])
    d = generate_array(d,.8)
    e = detect_triggerword(file,models[1])
    e = generate_array(e,.8)
    f = detect_triggerword(file,models[1])
    f = generate_array(f,.8)
    g = detect_triggerword(file,models[1])
    g = generate_array(g,.8)
    h = detect_triggerword(file,models[1])
    h = generate_array(h,.8)

    one = detect_triggerword(file,models[1])
    one = generate_array(one,.8)
    two = detect_triggerword(file,models[1])
    two = generate_array(two,.8)
    three = detect_triggerword(file,models[1])
    three = generate_array(three,.8)
    four = detect_triggerword(file,models[1])
    four = generate_array(four,.8)
    five = detect_triggerword(file,models[1])
    five = generate_array(five,.8)
    six = detect_triggerword(file,models[1])
    six = generate_array(six,.8)
    seven = detect_triggerword(file,models[1])
    seven = generate_array(seven,.8)
    eigth = detect_triggerword(file,models[1])
    eigth = generate_array(eigth,.8)



    return([record_m,a,b,c,d,e,f,g,h,one,two,three,four,five,six,seven,eigth])

# Calculate and plot spectrogram for a wav audio file
def my_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def detect_triggerword(filename,model):

    x = my_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    return predictions

def generate_array(predictions, threshold):
    result=[]
    flag_poss=False
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        
        #print(f"Predicción: {predictions[0,i,0]}")
        consecutive_timesteps +=1
        if predictions[0,i,0] > threshold:
            flag_poss = True
            #print("Flag poss True")
        
        if consecutive_timesteps > 75:
            consecutive_timesteps=0
            if flag_poss == True:
                result.append("1")
            else:
                result.append("-")
            flag_poss=False    
    return result 

######################################################################################
######################################################################################
######################################################################################

def clean_movements(first,lista,name):
    global record_m
    global a,b,c,d,e,f,g,h
    global one,two,three,four,five,six,seven,eigth
    #Transformo "." a "-"
    lista = [ "-" if ele=="." or ele=="-" else "1" for ele in lista  ]
    #Quito unos anteriores o que sucedan al mismo tiempo que el primer move
    lista=["-" if ind<first+1 else ele for ind,ele in enumerate(lista) ]
    #Para cada 1 quito las dos próximas ocurrencias donde hay 1
    #lista=quitar_duplicados(lista)
    #Quito todos los elementos de la lista que hayan ocurrido antes o al mismo tiempo que el primer move
    lista = [  ele for ind,ele in enumerate(lista)  if ind>first ]
    #Sustituo los 1 por el string que corresponde
    lista = populate_values(lista,name)

    if   name == "a":
        a=lista
    elif name == "b":
        b=lista
    elif name == "c":
        c=lista
    elif name == "d":
        d=lista
    elif name == "e":
        e=lista
    elif name == "f":
        f=lista
    elif name == "g":
        g=lista
    elif name == "h":
        h=lista
    elif name == "1":
        one=lista
    elif name == "2":
        two=lista
    elif name == "3":
        three=lista
    elif name == "4":
        four=lista
    elif name == "5":
        five=lista
    elif name == "6":
        six=lista
    elif name == "7":
        seven=lista
    elif name == "8":
        eigth=lista

    return lista

def get_movements( lista, name):
    #Generar arreglo raw
    lista_array_raw = generate_combinations(lista,name)
    #Generar combinaciones
    lista_comb_raw= list( itertools.product(*lista_array_raw) )
    
    #Generar movimientos
    lista_mov_raw =[]
    for i in lista_comb_raw:
        texto = "".join(i)
        for j in range(len(texto)-3):
            lista_mov_raw.append( texto[j:j+4] )
    
    #Generar movimientos validos
    lista_mov_val = [ele for ele in lista_mov_raw if ele[0] in ("a","b","c","d","e","f","g","h") \
                    and ele[2] in ("a","b","c","d","e","f","g","h") \
                    and ele[1] in ("1","2","3","4","5","6","7","8") \
                    and ele[3] in ("1","2","3","4","5","6","7","8")  ]
 
    return lista_mov_val

def quitar_duplicados (lista):
    ind_lista = [ i for i,ele in enumerate(lista) if ele=="1" ]
    for x in ind_lista:
        if ind_lista.index(x)< len(ind_lista)-1:
            if ind_lista[ ind_lista.index(x) ]+3>  ind_lista[ind_lista.index(x)+1]:
                del ind_lista[ind_lista.index(x)+1]
        if ind_lista.index(x)< len(ind_lista)-2:
            if ind_lista[ ind_lista.index(x) ]+3>  ind_lista[ind_lista.index(x)+1]:
                del ind_lista[ind_lista.index(x)+1]
    for x in ind_lista:
        lista = [ "-" if ind>x and ind<x+3 else ele for ind,ele in enumerate(lista) ]
    return lista

def populate_values(lista,nombre):
    lista = [nombre if ele=="1" else ele for ele in lista]
    return lista

def generate_combinations(lista,nombre):
    global record_m
    global a,b,c,d,e,f,g,h
    global one,two,three,four,five,six,seven,eigth
    #Obtener index del primer resultado de la lista que estoy recibiendo
    
    #Arreglo de arreglos
    total_arr=[]
    if nombre in lista:

        first = lista.index(nombre)
        #Arreglo por columna
        temp_arr =[]
        total_arr.append([nombre])
        
        if nombre in ("a","b","c","d","e","f","g","h"):

            #Ciclo desde la siguiente col(tiempo) del primer resultado hasta la última col(tiempo)    
            for i in range(first+1, len(lista) ):
                if one[i] != "-":
                    temp_arr.append(one[i])
                if two[i] != "-":
                    temp_arr.append(two[i])
                if three[i] != "-":
                    temp_arr.append(three[i])
                if four[i] != "-":
                    temp_arr.append(four[i])
                if five[i] != "-":
                    temp_arr.append(five[i])
                if six[i] != "-":
                    temp_arr.append(six[i])
                if seven[i] != "-":
                    temp_arr.append(seven[i])
                if eigth[i] != "-":
                    temp_arr.append(eigth[i])
                if a[i] != "-":
                    temp_arr.append(a[i])
                if b[i] != "-":
                    temp_arr.append(b[i])
                if c[i] != "-":
                    temp_arr.append(c[i])
                if d[i] != "-":
                    temp_arr.append(d[i])
                if e[i] != "-":
                    temp_arr.append(e[i])
                if f[i] != "-":
                    temp_arr.append(f[i])
                if g[i] != "-":
                    temp_arr.append(g[i])
                if h[i] != "-":
                    temp_arr.append(h[i])
                
                if len(temp_arr)>0:
                    total_arr.append(temp_arr)
                temp_arr=[]
                
        else:
            #Ciclo desde la siguiente col(tiempo) del primer resultado hasta la última col(tiempo)    
            for i in range(first+1, len(lista) ):
                if a[i] != "-":
                    temp_arr.append(a[i])
                if b[i] != "-":
                    temp_arr.append(b[i])
                if c[i] != "-":
                    temp_arr.append(c[i])
                if d[i] != "-":
                    temp_arr.append(d[i])
                if e[i] != "-":
                    temp_arr.append(e[i])
                if f[i] != "-":
                    temp_arr.append(f[i])
                if g[i] != "-":
                    temp_arr.append(g[i])
                if h[i] != "-":
                    temp_arr.append(h[i])
                if one[i] != "-":
                    temp_arr.append(one[i])
                if two[i] != "-":
                    temp_arr.append(two[i])
                if three[i] != "-":
                    temp_arr.append(three[i])
                if four[i] != "-":
                    temp_arr.append(four[i])
                if five[i] != "-":
                    temp_arr.append(five[i])
                if six[i] != "-":
                    temp_arr.append(six[i])
                if seven[i] != "-":
                    temp_arr.append(seven[i])
                if eigth[i] != "-":
                    temp_arr.append(eigth[i])
                
                
                if len(temp_arr)>0:
                    total_arr.append(temp_arr)
                temp_arr=[]
        
        
    return total_arr