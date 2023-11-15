from openai import OpenAI
import yaml
from pathlib import Path
from playsound import playsound

def speech2text(client: OpenAI, data_folder, file_name = "test.mp3"):
    audio_file= open(data_folder / file_name, "rb")
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

if __name__ == "__main__":
    system_prompt = "You are a helpful assistant"

    # Get the full filepath of the current script
    current_file = Path(__file__).resolve()
    current_directory = current_file.parent
    parant_directory = current_directory.parent
    data_folder = parant_directory / "data"
    print("Data folder: ", data_folder)

    # Open the YAML file
    with open(current_directory/'key.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Create the OpenAI client
    client = OpenAI(api_key=config['api_key'])

    # Get the transcript
    transcript = speech2text(client, data_folder)
    user_message = transcript
    print("User input: ",transcript)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )

    print("GPT output: ", response.choices[0].message.content)

    # Get the speech
    response, mp3_file_path = text2speech(client, data_folder, response.choices[0].message.content)
    playsound(mp3_file_path)