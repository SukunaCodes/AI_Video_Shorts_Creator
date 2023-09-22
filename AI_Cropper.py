from pytube import YouTube
import subprocess
import openai
import json
import math
import cv2
import pdb
import numpy as np

from youtube_transcript_api import YouTubeTranscriptApi

openai.api_key = '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'


# Download YouTube Video function
def download_youtube_video(url, filename):
    yt = YouTube(url)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video.download(filename=filename)


# Get segment video function
def segment_video(response):
    for i, segment in enumerate(response['segments']):
        start_time = math.floor(float(segment.get("start_time", 0)))
        end_time = math.ceil(float(segment.get("end_time", 0))) + 2
        output_file = f'output{str(i).zfill(3)}.mp4'
        command = f'ffmpeg -i input_video.mp4 -ss {start_time} -to {end_time} -c copy {output_file}'
        subprocess.call(command, shell=True)

# Face Detection function
def detect_faces(video_file):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load video
    cap = cv2.VideoCapture(video_file)
    faces = []

    # Detect and store unique faces
    while len(faces) < 5:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterate through detected faces
            for face in detected_faces:
                # Check if the face is already in the list of faces
                if not any(np.array_equal(face, f) for f in faces):
                    faces.append(face)

                # Print the number of faces detected so far
                print(f'Faces detected: {len(faces)}')

    # Release the VideoCapture object
    cap.release()

    # If faces detected, return the list of faces
    if len(faces) > 0:
        return faces

    # Otherwise return None
    return None
