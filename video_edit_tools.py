import cv2
from util import *


def get_npy_middle(vid_path, frames_wanted, npy_path):
    cap = cv2.VideoCapture(vid_path)

    frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    npy_file_num = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        start_frame = (frames_length // 2) - (frames_wanted // 2)
        end_frame = start_frame + frames_wanted
        # iterates over each frame in a video
        for frame_num in range(start_frame, end_frame):
            # sets the frame start for the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            # Read feed
            ret, frame = cap.read()
            # Make detections
            try:
                image, results = mediapipe_detection(frame, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # get key-points and save to .npy
                keypoints = extract_keypoints(results)
                # saves data key-points to .npy file
                np.save(os.path.join(npy_path, str(npy_file_num)), keypoints)
            except Exception as e:
                break
            npy_file_num += 1
    cap.release()


def pad_frames(vid_path, frames_wanted, npy_path):

    cap = cv2.VideoCapture(vid_path)
    frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        num_frames_to_insert = 60 - int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # List to store the new frames as NumPy arrays
        new_frames = []
        for frame_num in range(frames_wanted):
            # Read feed
            if frame_num < frames_length:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                # Read feed
                ret, frame = cap.read()
                # Make detections
                try:
                    image, results = mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    # get key-points and save to .npy
                    keypoints = extract_keypoints(results)
                    # saves data key-points to .npy file
                    np.save(os.path.join(npy_path, str(frame_num)), keypoints)
                except Exception as e:
                    break
            else:
                # Create a new frame by duplicating the last frame
                new_frame = keypoints
                np.save(os.path.join(npy_path, str(frame_num)), new_frame)

    # Release the input video
    cap.release()
