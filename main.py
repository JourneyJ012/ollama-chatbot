import ollama
from os import system
import pyttsx3
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import wave

ENGINE = pyttsx3.init()
whisper_model = WhisperModel(model_size_or_path="medium", device="cuda", compute_type="float16")

def list_mics():
    devices = sd.query_devices()
    return [device['name'] for device in devices]

def choose_mic(mic_list):
    choice = int(input("Enter the number corresponding to the mic you want to use: "))
    return mic_list[choice]

def record_audio(duration=5, fs=16000, mic_index=None):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16, device=mic_index)
    sd.wait() 
    print("Recording complete.")

    with wave.open('temp_audio.wav', 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2) 
        f.setframerate(fs)
        f.writeframes(audio_data.tobytes())
    
    return 'temp_audio.wav'

def listen_and_transcribe(mic_index, model: WhisperModel):
    audio_file = record_audio(mic_index=mic_index)
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "
    
    print("You said: " + transcribed_text.strip())
    return transcribed_text.strip()

def speak(text, engine):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":

    names = [model['name'] for model in ollama.list()['models']]
    for i, item in enumerate(names):
        print(i, item)

    ollama_model = names[int(input("Enter the number corresponding to the model you want to use: "))]

    system('cls')

    mic_list = list_mics()
    for i, item in enumerate(mic_list):
        print(i, item)
    mic_name = choose_mic(mic_list)

    system('cls')
    while True:
        mic_index = mic_list.index(mic_name)
        
        text = listen_and_transcribe(mic_index, whisper_model)
        if text not in ["", " "]:
            if 'messages' not in globals():
                messages = [
                    {'role': 'system', 'content': 'You are a voice, responding to mic input. It may be inaccurate, so use previous messages to try predict what the user is talking about. Do not add any markdown punctuation like asterisks. When possible, answer in less than 30 words.'},
                    {'role': 'user', 'content': text}
                ]
            else:
                messages.append({'role': 'user', 'content': text})

            response = ollama.chat(model=ollama_model, messages=messages)
            speak(text=response['message']['content'], engine=ENGINE)
            messages.append({'role': 'assistant', 'content': response['message']['content']})
