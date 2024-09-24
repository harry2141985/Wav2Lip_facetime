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

# Load models (text to speech and speech to text and llm)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
model_name = 'microsoft/phi-2'
generator = pipeline("text-generation", model=model_name, device_map='cuda')  # Change to 'cpu' if needed
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

def perform_lip_sync(checkpoint_path, face, audio, outfile):
    # Prepare arguments for lip-sync
    args = argparse.Namespace()
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

def chat_with_emma(user_input, checkpoint_path, face_filepath, audio_filepath):
    conv_hist = ''
    response = ''
    
    # Update conversation history
    conv_hist += f"\nYou: {user_input}\n"
    
    # Model response generation
    full_input = demo + conv_hist
    response = generator(full_input, max_new_tokens=25, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)[0]['generated_text']
    
    # Process response
    response = response.replace(full_input, '').strip()
    conv_hist += f"Emma: {response}\n"
    
    text_wav = response.replace("Emma: ", "").strip()
    filepath = f"temp_audio.wav"  # Temporary audio file path
    
    # Text to speech
    print("Generating voice")
    tts.tts_to_file(text=text_wav, language="en", speaker_wav="iann.wav", file_path=filepath)
    print("Playing audio:")
    
    # Uncomment to play the generated audio
    # play(AudioSegment.from_file(filepath))

    # Perform lip-syncing
    outfile = "results/result_voice.mp4"  # Output file for the lip-synced video
    perform_lip_sync(checkpoint_path, face_filepath, filepath, outfile)

    # Return the result video path and conversation history
    return outfile, conv_hist

if __name__ == '__main__':
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
    parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str, help='Filepath of audio file to use for lip syncing', required=True)  
    args = parser.parse_args()

    # Set up Gradio interface with sharing enabled
    iface = gr.Interface(
        fn=lambda user_input: chat_with_emma(user_input, args.checkpoint_path, args.face, args.audio),
        inputs="text",
        outputs=["video", "text"],
        title="Chat with Emma",
        description="Interact with Emma, a friendly florist, and get flower recommendations!"
    )

    # Launch the interface with share=True
    iface.launch(share=True)
