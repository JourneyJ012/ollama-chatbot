import speech_recognition as sr
import ollama
from os import system
import pyttsx3

ENGINE = pyttsx3.init()

def list_mics():
    return sr.Microphone.list_microphone_names()

def choose_mic(mic_list):
    #I previously had this in a try statement, but it should return an error if something goes wrong anyways.
    choice = int(input("Enter the number corresponding to the mic you want to use: "))
    return mic_list[choice]

def listen_and_transcribe(mic): #two in one is easier :D
    with sr.Microphone(device_index=mic) as source:
        print("Say something!")
        audio = r.listen(source)
        try:
            text = r.recognize_sphinx(audio)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, the Speech Recognition could not understand what was passed through.")
        return None
    
def speak(text, engine):
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":

    
    names = [model['name'] for model in ollama.list()['models']]
    for i, item in enumerate(names):
        print(i, item)

    model = names[int(input("Enter the number corresponding to the model you want to use:    "))]
    #let this raise an error. 

    system('cls')

    mic_list = list_mics()
    for i, item in enumerate(mic_list):
        print(i, item)
    mic_name = choose_mic(mic_list)
    
    system('cls')
    while True:
        r = sr.Recognizer()
        mic_index = mic_list.index(mic_name)
        
        text = listen_and_transcribe(mic_index)
        if text not in [""," "]:
            # Initialize the conversation history if it hasn't been done already
            if 'messages' not in globals():
                messages = [
                    {'role': 'system', 'content': 'You are a voice, responding to mic input. It may be innaccurate, so use previous messages to try predict what the user is talking about. Do not add any markdown punctuation like asterisks. When possible, answer in less than 30 words'},
                    {'role': 'user', 'content': text}]
            else:
                messages.append({'role': 'user', 'content': text})

            
            response = ollama.chat(model=model, messages=messages)
            speak(text=response['message']['content'],engine=ENGINE)
            messages.append({'role': 'assistant', 'content': response['message']['content']})