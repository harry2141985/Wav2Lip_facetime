from TTS.api import TTS
from transformers import pipeline, AutoTokenizer
import whisper
from scipy.io.wavfile import write
from playsound import playsound
import sounddevice as sd
import requests
import pandas as pd
import json
import re
import os
import threading
import asyncio
import cv2
import time
from Wav2lip_inference import main
import gradio as gr
from pydub import AudioSegment
from pydub.playback import play
import argparse
import multiprocessing

# Load models (text to speech and speech to text and llm)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
model_name = 'microsoft/phi-2'
model_name_or_path = "microsoft/phi-2"  # Define the model name or path
generator = pipeline("text-generation", model=model_name_or_path, device_map='cuda')  # Change to 'cpu' if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define Emma's persona
demo = """Emma's persona: Emma is a young woman who works in a charming flower shop. She has a passion for flowers and enjoys helping customers choose the perfect arrangements. Emma is friendly and loves to share her knowledge about different flower types and their meanings. She also enjoys gardening in her spare time and is known for her cheerful personality. Emma's full name is Emma Bloom. She is 24 years old and loves indie music.
<START>
[DIALOGUE HISTORY]
Emma: Hello there! Welcome to our flower shop. How can I help you today?
You: Hi Emma! I’m looking for some flowers for a special occasion.
Emma: That sounds wonderful! What occasion is it for? You can tell me about it, and I'll help you pick the perfect flowers!
Emma: Great! We have some lovely options. Would you like something bright and cheerful, or something elegant and sophisticated?"""

print('demo\n')
print(demo)
output_file_path = "passit_output.txt"
with open(output_file_path, "w") as file:
    file.write(demo)

def perform_lip_sync(checkpoint_path, face, audio, outfile):
    args.checkpoint_path = checkpoint_path
    args.face = face
    args.audio = audio
    args.outfile = outfile
    args.resize_factor = 1
    args.rotate = False  
    args.crop = [0, -1, 0, -1]
    args.wav2lip_batch_size = 128
    args.resize_face = 1
    main(args)

if __name__ == '__main__':
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result', default='results/result_voice.mp4') 
    args = parser.parse_args()

    conv_hist = ''
    user_input = 'new response'
    response = ''

    for i in range(0, 10):
        newinput = input("Type your response to Emma: ")
        user_input = "\nYou: " + newinput
        
        # Update conversation history
        conv_hist += user_input + '\n'
        
        # Model response generation
        user_input = demo + conv_hist
        response = generator(user_input, max_new_tokens=25, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)[0]['generated_text']
        
        # Process response
        response = response.replace(user_input, '').strip()
        conv_hist += "Emma: " + response + '\n'
        
        text_wav = response.replace("Emma: ", "").strip()
        filepath = f"{i}.wav"
        
        # Text to speech
        print("Generating voice")
        tts.tts_to_file(text=text_wav, language="en", speaker_wav="iann.wav", file_path=filepath)
        print("Playing audio:")
        
        # Uncomment to play the generated audio
        # play(AudioSegment.from_file(filepath))

        # Perform lip-syncing
        face_filepath = "ian5sec25fps.mp4"  # Path to the video/image with face
        checkpoint_filepath = args.checkpoint_path  # Use command-line argument for checkpoint
        outfile = args.outfile  # Output file for the lip-synced video
        perform_lip_sync(checkpoint_filepath, face_filepath, filepath, outfile)

        # Update the Gradio interface after completing the loop
        passit = demo + conv_hist
        output_file_path = "passit_output.txt"

        # Write the content of the 'passit' variable to the text file
        with open(output_file_path, "w") as file:
            file.write(passit)
