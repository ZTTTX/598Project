from openai import OpenAI
import yaml
from pathlib import Path
from playsound import playsound
from typing import Optional
from transcribe import wav_to_text, text_to_speech_mac

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def speech2text(client: OpenAI, audio_path):
    audio_file= open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file,
        response_format="text"
    )

    return transcript

def text2speech(client: OpenAI, data_folder, text: str = "Hello, World!"):
    speech_file_path = data_folder / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(speech_file_path)
    return response, speech_file_path

def one_step_conversation(message_history: list, emotion: str = "neutral", audio_path = None):
    # Set up directories
    current_file = Path(__file__).resolve()
    current_directory = current_file.parent
    parant_directory = current_directory.parent
    data_folder = parant_directory / "data"
    if audio_path is None:
        audio_path = data_folder / "test.mp3"

    # Create the client
    with open(current_directory/'key.yaml', 'r') as file:
        config = yaml.safe_load(file)
    client = OpenAI(api_key=config['api_key'])

    if message_history == []:
        # Initialize the message with system prompt
        system_prompt = file_to_string(current_directory / "system_prompt.txt")
        message = {"role": "system", "content": system_prompt}
        message_history.append(message)

    # Add user message to the message history
    user_prompt = file_to_string(current_directory / "user_prompt.txt")
    # Get the transcript
    transcript = speech2text(client, audio_path)
    user_message = {"role": "user", "content": user_prompt.format(EMOTION=emotion, DIALOGUE=transcript)}
    message_history.append(user_message)
    print("User emotion: ",emotion)
    print("User message: ",transcript)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    print("GPT output: ", response.choices[0].message.content)
    assistant_message = {"role": "assistant", "content": response.choices[0].message.content}
    message_history.append(assistant_message)

    # Get the speech
    response, mp3_file_path = text2speech(client, data_folder, response.choices[0].message.content)
    playsound(mp3_file_path)

    return message_history

def one_step_conversation_sr(message_history: list, emotion: str = "neutral", audio_path = None):
    # Set up directories
    current_file = Path(__file__).resolve()
    current_directory = current_file.parent
    parant_directory = current_directory.parent
    data_folder = parant_directory / "data"
    if audio_path is None:
        audio_path = data_folder / "test.mp3"

    # Create the client
    with open(current_directory/'key.yaml', 'r') as file:
        config = yaml.safe_load(file)
    client = OpenAI(api_key=config['api_key'])

    if message_history == []:
        # Initialize the message with system prompt
        system_prompt = file_to_string(current_directory / "system_prompt.txt")
        message = {"role": "system", "content": system_prompt}
        message_history.append(message)

    # Add user message to the message history
    user_prompt = file_to_string(current_directory / "user_prompt.txt")
    # Get the transcript
    transcript = wav_to_text(audio_path)
    user_message = {"role": "user", "content": user_prompt.format(EMOTION=emotion, DIALOGUE=transcript)}
    message_history.append(user_message)
    print("User emotion: ",emotion)
    print("User message: ",transcript)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    print("GPT output: ", response.choices[0].message.content)
    assistant_message = {"role": "assistant", "content": response.choices[0].message.content}
    message_history.append(assistant_message)

    # Get the speech
    speech_text = response.choices[0].message.content
    text_to_speech_mac(speech_text)

    return message_history