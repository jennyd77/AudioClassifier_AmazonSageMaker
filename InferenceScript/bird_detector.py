#python3
import boto3
import botocore
import json
import requests
import time
import os.path
import sys
import getopt
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
# pip3 install numpy
# pip3 install pydub --user
# pip3 install librosa
# pip3 install matplotlib
from pydub import AudioSegment
from datetime import datetime
inference_categories = ['background', 'curlew', 'whipbird']

client = boto3.client('sagemaker-runtime', region_name='ap-southeast-2')
endpoint_name = 'birdsong-pre-mel-epoch10-minibatch8-ep--2022-03-16-03-41-52'

def run_inference(file_name):
    # Code to run inference
    with open(file_name, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)
    response = client.invoke_endpoint(EndpointName=endpoint_name,
                                   ContentType='application/x-image',
                                   Body=payload)
    result = response['Body'].read()
    result = json.loads(result)
    if args.verbose: print(result)
    index = np.argmax(result)
    if args.verbose: print(index)
    confidence = result[index]
    return inference_categories[index], confidence

def make_melspgram(audio_segment):

    # Attempt 1 to convert audio_segment to Librosa array in memory
    #samples = audio_segment.get_array_of_samples()
    #samples_array = np.array(samples).astype(np.float32)/32768
    #array = librosa.core.resample(samples_array,audio_segment.frame_rate, 22050, res_type='kaiser_best')

    # Attempt 2 to convert audio_segment to Librosa array in memory
    #channel_sounds = audio_segment.split_to_mono()
    #samples = [s.get_array_of_samples() for s in channel_sounds]

    #fp_arr = np.array(samples).T.astype(np.float32)
    #fp_arr /= np.iinfo(samples[0].typecode).max
    #fp_arr = fp_arr.reshape(-1)

    #if args.verbose: print("Resample in memory Librosa array {}".format(fp_arr))

    # Saving audio segment as a file in order to load to Librosa.
    # If someone knows how to do this in memory, please submit a pull request
    audio_segment.export('/tmp/audio_segment.wav', format="wav")
    clip, sample_rate = librosa.load('/tmp/audio_segment.wav', sr=None)
    #if args.verbose: print("Load from file Librosa array {}".format(clip))

    n_mels = 64
    #base = os.path.basename(file_path)
    #input_file = os.path.splitext(base)[0]

    n_fft = 1024 # frame length

    fmin = 0
    fmax = 22050 # sample_rate/2
    hop_length = 512
    mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length,
                                              n_mels=n_mels, sr=sample_rate, power=1.0,
                                              fmin=fmin, fmax=fmax)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spec_db, x_axis=None,  y_axis=None,
                             sr=sample_rate, hop_length=hop_length,
                             fmin=fmin, fmax=fmax)
    path_to_melsgplt='/tmp/melsgplt.png'
    plt.savefig(path_to_melsgplt)
    return path_to_melsgplt

def mono_to_stereo(AudioFile):

    left_channel = AudioFile
    right_channel = AudioFile

    stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
    #file_no_ext = os.path.splitext(file_path)[0]
    #stereo_sound.export(file_no_ext+'_stereo.wav', format="wav")
    #return file_no_ext+'_stereo.wav'
    return stereo_sound

def process_audio(audio_file):
    predictions = {}
    length = audio_file.duration_seconds
    if args.verbose: print("audio file length {}".format(length))
    for i in range(0, (int(length)-3)*1000, 500):
        json_entry = {}
        start = i
        end = i+3000
        if args.verbose: print("start {}, end {}".format(start,end))
        audio_segment = audio_file[start:end]
        path_to_image = make_melspgram(audio_segment)
        prediction, confidence = run_inference(path_to_image)
        print(prediction, confidence)

        json_entry["timestamp_start"] = start
        json_entry["timestamp_end"] = end
        json_entry["prediction"] = prediction
        json_entry["confidence"] = confidence
        predictions[start]=json_entry
        #predictions.append(json_entry)
    return json.dumps(predictions)


region = 'ap-southeast-2'
s3bucket = 'djenny-appservices-demos'
s3directory = 'transcribe'
content_dir = 'content'

help_msg = "This program uses an Amazon SageMaker endpoint to detect bird species in an audio file"
parser = argparse.ArgumentParser(description = help_msg)
parser.add_argument("input_file", type=str, help="audio input filename")
parser.add_argument("-V", "--verbose", help="run in verbose output mode", action="store_true")
args=parser.parse_args()

input_file = args.input_file

if args.verbose:
    print("File to scan: ",input_file)

base_dir = os.path.dirname(input_file)
base_filename = Path(input_file).stem
file_ext = os.path.splitext(input_file)[1]
file_ext_final = file_ext.replace('.', '')
new_filename=os.path.join(base_dir,base_filename) + '.wav'
key_t=os.path.join(s3directory, new_filename)

if args.verbose:
    print("base_dir: ",base_dir)
    print("base_filename: ",base_filename)
    print("file_ext_final: ",file_ext_final)
    print("new_filename: ",new_filename)
    print("key_t: ",key_t)

# If not wav, convert now
if (file_ext_final != 'wav'):
   print("Converting " + input_file + " to wav format")
   try:
      track = AudioSegment.from_file(input_file, file_ext_final)
      file_handle = track.export(new_filename, format='wav')
   except:
      print("ERROR: Could not convert " + str(filepath) + " to wav format")
else:
   print("Input file is already wav format. Proceeding directly to transcription")

# Convert mono to stereo (unless already stereo)
audio_file = AudioSegment.from_file(new_filename)
channel_count = audio_file.channels
if args.verbose: print("Channel count {}".format(channel_count))
if channel_count == 1:
    audio_file = mono_to_stereo(audio_file)

# Slice audio and run inference on each slice
response = process_audio(audio_file)
print(response)
