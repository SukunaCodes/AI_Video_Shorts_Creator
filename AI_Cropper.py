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


# Crop video function
def crop_video(faces, input_file, output_file):
    try:
        if len(faces) > 0:
            # Constants for cropping
            CROP_RATIO = 0.9
            VERTICAL_RATIO = 9 / 16

            # Read input video
            cap = cv2.VideoCapture(input_file)

            # Get the frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate target width and height for cropping (vertical format)
            target_width = int(frame_height * VERTICAL_RATIO)
            target_height = int(frame_height * CROP_RATIO)

            # Create VideoWriter object to save the cropped video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(output_file, fourcc, 30, (target_width, target_height))

            # Loop each frame of the output video
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Iterate through each detected face
                for face in faces:
                    # Get the coordinates and dimensions of the face
                    x, y, w, h = face

                    # Calculate the coordinates of the cropped frame
                    crop_x = max(0, x + (w - target_width) // 2)  # Adjust the crop region to center the face
                    crop_y = max(0, y + (h - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    # Crop the frame based on the calculated crop coordinates
                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]

                    # Resize the cropped frame to the target dimensions
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

                    # Write the resized frame to the output video
                    output_video.write(resized_frame)

            # Release the VideoCapture and VideoWriter objects
            cap.release()
            output_video.release()
            print('Video cropped successfully.')
        else:
            print('No faces detected.')
    except Exception as e:
        print(f'Error cropping video: {str(e)}')


def crop_video2(faces, input_file, output_file):
    try:
        if len(faces) > 0:
            # Constants for cropping
            CROP_RATIO = 0.9  # Adjust the ratio to control how much of the face is visible in the cropped video
            VERTICAL_RATIO = 9 / 16  # Aspect ratio for the vertical video
            BATCH_DURATION = 5  # Duration of each batch in seconds

            # Read the input video
            cap = cv2.VideoCapture(input_file)

            # Get the frame dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate the target width and height for cropping (vertical format)
            target_height = int(frame_height * CROP_RATIO)
            target_width = int(target_height * VERTICAL_RATIO)

            # Calculate the number of frames per batch
            frames_per_batch = int(cap.get(cv2.CAP_PROP_FPS) * BATCH_DURATION)

            # Create a VideoWriter object to save the output video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(output_file, fourcc, 30.0, (target_width, target_height))

            # Loop through each batch of frames
            while True:
                ret, frame = cap.read()

                # If no more frames, break out of the loop
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert frame to BGR color format
                # Iterate through each detected face
                for face in faces:
                    # Unpack the face coordinates
                    x, y, w, h = face

                    # Calculate the crop coordinates
                    crop_x = max(0, x + (w - target_width) // 2)  # Adjust the crop region to center the face
                    crop_y = max(0, y + (h - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    # Crop the frame based on the calculated crop coordinates
                    cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]

                    # Resize the cropped frame to the target dimensions
                    resized_frame = cv2.resize(cropped_frame, (target_width, target_height))

                    # Write the resized frame to the output video
                    output_video.write(resized_frame)

                    # Check if the current frame index is divisible by frames_per_batch
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) % frames_per_batch == 0:
                        # Analyze the lip movement or facial muscle activity within the batch
                        is_talking = is_talking_in_batch(resized_frame)

                        # Adjust the focus based on the speaking activity
                        adjust_focus(is_talking)

            # Release the input and output video objects
            cap.release()
            output_video.release()

            print("Video cropped successfully.")
        else:
            print("No faces detected in the video.")
    except Exception as e:
        print(f"Error during video cropping: {str(e)}")


def is_talking_in_batch(frames):
    # Calculate motion btwn consecutive frames
    motion_scores = []
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        motion_score = calculate_motion_score(frame1, frame2)  # Replace with your motion analysis function
        motion_scores.append(motion_score)

        # Determine if talking behavior is present based on motion scores
    threshold = 0.5  # Adjust the threshold as needed
    talking = any(score > threshold for score in motion_scores)

    return talking


def calculate_motion_score(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude of optical flow vectors
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Calculate motion score as the average magnitude of optical flow vectors
    motion_score = np.mean(magnitude)

    return motion_score


def adjust_focus(frame, talking):
    if talking:
        # Apply visual effects or adjustments to emphasize the speaker
        # For example, you can add a bounding box or overlay text on the frame
        # indicating the speaker is talking
        # You can also experiment with resizing or positioning the frame to
        # focus on the talking person

        # Example: Draw a bounding box around the face region
        face_coordinates = get_face_coordinates(frame)  # Replace with your face detection logic

        if face_coordinates is not None:
            x, y, w, h = face_coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def get_face_coordinates(frame):
    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Return the coordinates of the first detected face
        x, y, w, h = faces[0]
        return x, y, w, h

    # If no face detected, return None
    return None


def get_transcript(video_id):
    # Get the transcript for the given YouTube video ID
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Format the transcript for feeding into GPT-4
    formatted_transcript = ''
    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return transcript


# Analyze transcript with GPT-3 function
response_obj = '''[
  {
    "start_time": 97.19, 
    "end_time": 127.43,
    "description": "Spoken Text here"
    "duration":36 #Length in seconds
  },
  {
    "start_time": 169.58,
    "end_time": 199.10,
    "description": "Spoken Text here"
    "duration":33 
  },
]'''


def analyze_transcript(transcript):
    prompt = f"This is a transcript of a video. Please identify the 3 most viral sections from the whole, make sure they are more than 30 seconds in duration,Make Sure you provide extremely accurate timestamps respond only in this format {response_obj}  \n Here is the Transcription:\n{transcript}"
    messages = [
        {"role": "system",
         "content": "You are a ViralGPT helpful assistant. You are master at reading youtube transcripts and identifying the most Interesting and Viral Content"},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=512,
        n=1,
        stop=None
    )
    return response.choices[0]['message']


def main():
    video_id = '92nse3cvG_Y'
    url = 'https://www.youtube.com/watch?v=' + video_id  # Replace with your video's URL
    filename = 'input_video.mp4'
    download_youtube_video(url, filename)

    transcript = get_transcript(video_id)
    print(transcript)
    interesting_segment = analyze_transcript(transcript)
    print(interesting_segment)
    content = interesting_segment["content"]
    parsed_content = json.loads(content)
    print(parsed_content)
    # pdb.set_trace()
    segment_video(parsed_content)

    # Loop through each segment
    for i in range(0, 3):  # Replace 3 with the actual number of segments
        input_file = f'output{str(i).zfill(3)}.mp4'
        output_file = f'output_cropped{str(i).zfill(3)}.mp4'
        faces = detect_faces(input_file)
        crop_video(faces, input_file, output_file)

    # Assume you have a way to get the transcript. This is not shown here.


# Replace with actual transcript


# Run the main function
main()