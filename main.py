from data_capture import capture_audio_video
from emotion_detection import from_video_get_emotion, detect_main_emotion
from gpt_api.gpt_api import one_step_conversation

if __name__ == "__main__":
    # set up directori
    video_path = "./data_capture/video.mp4"
    audio_path = "./data_capture/audio.wav"
    capture_time = 5

    # Initialize message history
    message_history = []

    # While loop for the conversation
    while True:
        # Capture audio and video
        capture_audio_video(capture_time, video_path, audio_path)

        # Get emotion from video
        emotions = from_video_get_emotion(video_path, num_frames=5)
        main_emotion = detect_main_emotion(emotions)

        # Get response from GPT-3
        message_history = one_step_conversation(message_history, main_emotion, audio_path)

        # Print message history just for debugging
        print(message_history)