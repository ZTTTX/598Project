import os
import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
import matplotlib.pyplot as plt
from deepface import DeepFace

def capture_audio(duration, filepath):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def capture_video(duration, filepath):
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using 'mp4v' codec
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (1280, 720), True)

    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def capture_audio_video(duration, video_path="output.avi", audio_path="output.mp4"):
    audio_thread = threading.Thread(target=capture_audio, args=(duration, audio_path))
    video_thread = threading.Thread(target=capture_video, args=(duration, video_path))

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = np.linspace(30, total_frames - 1, num=num_frames, dtype=int)

    extracted_frames = []

    for frame_num in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            extracted_frames.append(frame)

    cap.release()
    return extracted_frames

def display_frames(frames):
    plt.figure(figsize=(15, 10))
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()