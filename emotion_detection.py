from deepface import DeepFace
from data_capture import extract_frames

def get_dominant_emotion(frame):
    """
    Analyzes a single frame and returns the dominant emotion.
    
    :param frame: A single image frame (numpy array).
    :return: Dominant emotion as a string. Returns None if no face is detected or an error occurs.
    """
    try:
        # Analyze the frame for emotions
        analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
        
        # Extract the dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print("Error in processing frame:", e)
        return None
    

def from_video_get_emotion(video_path, num_frames=5):
    frames = extract_frames(video_path, num_frames=5)
    emotions = []
    for frame in frames:
        emotions.append(get_dominant_emotion(frame))
        
    return emotions

def detect_main_emotion(emotions):
    # Analyze the most frequent emotion
    if emotions:
        most_common_emotion = max(set(emotions), key=emotions.count)
    else:
        most_common_emotion = "neutral"
    
    return most_common_emotion