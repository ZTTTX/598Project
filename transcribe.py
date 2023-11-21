import SpeechRecognition as sr
import os

def transcribe_audio(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def wav_to_text(wav_file):
    text = transcribe_audio(wav_file)
    return text

def text_to_speech_mac(text):
    os.system(f'say "{text}"')