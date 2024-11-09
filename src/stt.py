from faster_whisper import WhisperModel
import numpy as np
import wave
import sounddevice as sd

def list_mics():
    """Lists available microphones."""
    devices = sd.query_devices()
    return [device['name'] for device in devices]

def choose_mic(mic_list):
    """Prompts user to choose a microphone."""
    choice = int(input("Enter the microphone number: "))
    return mic_list[choice]

def record_audio(silence_threshold=500, silence_duration=1.0, fs=16000, mic_index=None):
    """Records audio from microphone until silence is detected."""
    print("Recording...")
    buffer = []
    silence_samples = int(silence_duration * fs)
    silent_chunks = 0
    has_speech = False 

    while True:
        audio_chunk = sd.rec(int(0.1 * fs), samplerate=fs, channels=1, dtype=np.int16, device=mic_index)
        sd.wait()
        
        buffer.append(audio_chunk)
        amplitude = np.max(np.abs(audio_chunk))
        
        if amplitude < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
            has_speech = True 
        
        if silent_chunks * 0.1 >= silence_duration:
            print("Recording complete.")
            break

    if not has_speech:
        print("No speech detected.")
        return None

    audio_data = np.concatenate(buffer, axis=0)
    with wave.open('temp_audio.wav', 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(fs)
        f.writeframes(audio_data.tobytes())
    
    return 'temp_audio.wav'

def listen_and_transcribe(mic_index, model: WhisperModel):
    """Records and transcribes audio using Whisper."""
    audio_file = record_audio(mic_index=mic_index)
    
    if audio_file is None:
        return "" 

    segments, info = model.transcribe(audio_file, beam_size=5)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    transcribed_text = " ".join([segment.text for segment in segments])
    print("You said: " + transcribed_text.strip())
    
    return transcribed_text.strip()
