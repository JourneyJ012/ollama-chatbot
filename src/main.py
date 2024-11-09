import ollama
import pyttsx3
from os import system
import stt
from faster_whisper import WhisperModel


ENGINE = pyttsx3.init()

def speak(text, engine):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    
    models = ollama.list()['models']
    model_names = [model['name'] for model in models]
    for i, name in enumerate(model_names):
        print(f"{i}: {name}")
    ollama_model = model_names[int(input("Enter the model number: "))]
    system('cls')

    
    mic_list = stt.list_mics()
    for i, mic in enumerate(mic_list):
        print(f"{i}: {mic}")
    mic_name = stt.choose_mic(mic_list)
    system('cls')

    
    whisper_model = WhisperModel("medium", device="cuda", compute_type="int8")  

    messages = [
        {'role': 'system', 'content': (
            'You are a voice assistant. Respond to microphone input. Responses should be concise, avoiding markdown and staying under 30 words when possible.'
        )}
    ]

    
    while True:
        mic_index = mic_list.index(mic_name)
        text = stt.listen_and_transcribe(mic_index, whisper_model)

        if text:
            messages.append({'role': 'user', 'content': text})
            response = ollama.chat(model=ollama_model, messages=messages)

            speak(response['message']['content'], ENGINE)
            messages.append({'role': 'assistant', 'content': response['message']['content']})
