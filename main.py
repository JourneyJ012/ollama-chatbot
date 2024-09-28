import speech_recognition as sr
import ollama
from os import system
import pyttsx3
from faster_whisper import WhisperModel

ENGINE = pyttsx3.init()
whisper_model = WhisperModel(model_size_or_path="medium", device="cuda", compute_type="float16")

def list_mics():
    return sr.Microphone.list_microphone_names()

def choose_mic(mic_list):
    #I previously had this in a try statement, but it should return an error if something goes wrong anyways.
    choice = int(input("Enter the number corresponding to the mic you want to use: "))
    return mic_list[choice]

def listen_and_transcribe(mic, model: WhisperModel):
    r = sr.Recognizer()
    with sr.Microphone(device_index=mic) as source:
        print("Say something!")
        audio = r.listen(source)

        try:
            # Save the audio as a temp file
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())

            segments, info = model.transcribe("temp_audio.wav", beam_size=5)
            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
            
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + " "  #Combine all segments into a single text
            print("You said: " + transcribed_text.strip()) 
            return transcribed_text.strip()

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

    ollama_model = names[int(input("Enter the number corresponding to the model you want to use:    "))]
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
        
        text = listen_and_transcribe(mic_index, whisper_model)
        if text not in [""," "]:
            # Initialize the conversation history if it hasn't been done already
            if 'messages' not in globals():
                messages = [
                    {'role': 'system', 'content': 'You are a voice, responding to mic input. It may be innaccurate, so use previous messages to try predict what the user is talking about. Do not add any markdown punctuation like asterisks. When possible, answer in less than 30 words'},
                    {'role': 'user', 'content': text}]
            else:
                messages.append({'role': 'user', 'content': text})

            
            response = ollama.chat(model=ollama_model, messages=messages)
            speak(text=response['message']['content'],engine=ENGINE)
            messages.append({'role': 'assistant', 'content': response['message']['content']})