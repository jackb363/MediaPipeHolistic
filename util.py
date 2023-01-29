import cv2
import numpy as np
import mediapipe as mp
import os
from threading import Thread
from natsort import natsorted
from multiprocessing import Process

# Holistic model
mp_holistic = mp.solutions.holistic
# Drawing utilities
mp_drawing = mp.solutions.drawing_utils


# function to call mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


# function uses mediapipe to extract keypoints from frames
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


# function to superimpose keypoints over frame
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


# function that gets only files ignores folders
def get_files(path):
    file_names = list()
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            file_names.append(file)
    return file_names


# function to pad .npy files to a specified number
def pad_npy(npy_dir, current_files, new_file_max):
    first_file = np.load(os.path.join(npy_dir, "0.npy"))
    # loop to create new .npy files and copy them to a directory
    for copy_frame in range(current_files, new_file_max):
        padded_file = first_file
        npy_path = os.path.join(npy_dir, str(copy_frame))
        np.save(npy_path, padded_file)


# function takes the middle number of .npy files specified
def cut_npy(npy_dir, cut_to_num):
    file_list = natsorted(os.listdir(npy_dir))
    last_file = extract_number(file_list[-1])
    to_remove_num = (last_file - cut_to_num) / 2
    if to_remove_num.is_integer():
        remove_files(npy_dir, 0, int(to_remove_num))
        remove_files(npy_dir, last_file-int(to_remove_num), last_file+1)
    else:
        remove_files(npy_dir, 0, round(to_remove_num))
        remove_files(npy_dir, last_file - int(to_remove_num), last_file+1)


# removes the files within a range
def remove_files(dir_path, start_file, end_file):
    for filename in range(start_file, end_file):
        if os.path.exists(os.path.join(dir_path, str(filename) + '.npy')):
            os.remove(os.path.join(dir_path, str(filename) + '.npy'))


# deletes all files in specified directory
def delete_files(dir_path):
    current_dir = os.path.split(dir_path)
    print('Deleting Files For Directory', current_dir[1])
    for file in os.scandir(dir_path):
        os.remove(file.path)


def extract_number(file_name):
    return int(os.path.splitext(file_name)[0])


# runs tasks in parallel
def run_parallel(*functions):
    threads = []
    for function in functions:
        thread = Thread(target=function)
        thread.start()
        threads.append(thread)
    for t in threads:
        t.join()


def resize_video(video_path, desired_frames):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Check if the video has more or less than the desired number of frames
    if total_frames > desired_frames:
        print('more than needed')
        # Get the frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the new frame rate
        new_fps = fps * (total_frames / desired_frames)
        # Create a new video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('resized_' + video_path, fourcc, new_fps, (frame_width, frame_height))
        # Read the frames and write the resized video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.resize(frame, (frame_width, frame_height)))
        # Release the resources
        cap.release()
        out.release()
    elif total_frames < desired_frames:
        # Get the frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculate the new frame rate
        new_fps = fps * (desired_frames / total_frames)
        # Create a new video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('resized_' + video_path, fourcc, new_fps, (frame_width, frame_height))
        # Read the frames and write the resized video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.resize(frame, (frame_width, frame_height)))
            for i in range(int(new_fps / fps) - 1):
                out.write(frame)
        # Release the resources
        cap.release()
        out.release()
    else:
        print("The number of frames in the video is already the desired number.")