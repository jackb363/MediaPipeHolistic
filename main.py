from util import *
import os
import mediapipe as mp
from video_edit_tools import *
# path to dataset and file list
root_dir = 'C:/Users/Jack/Documents/MediaPipe_SmallDataset'
root_dir_files = os.listdir(root_dir)

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def delete_npy_files(npy_file_dir):
    for folder in npy_file_dir:
        vid_list = get_files(os.path.join(root_dir, folder))
        # iterates over each video in category
        for vid in vid_list:
            print("Beginning File Deletion For", vid)
            npy_dir = os.path.join(root_dir, folder, str(vid_list.index(vid)))
            delete_files(npy_dir)
            print("Finished File Deletion For", vid + '\n')


def edit_video_frame_num(frames_wanted):
    # mediapipe is used to extract keypoints from the video
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # iterates over the categories in dataset
        for folder in root_dir_files:
            # gets all videos in a category
            vid_list = get_files(os.path.join(root_dir, folder))
            # iterates over each video in category
            for vid in vid_list:
                # creates dir for .npy files
                os.mkdir(os.path.join(root_dir, folder, str(vid_list.index(vid))))
                # loads current vid to videocapture to check frame num
                cap = cv2.VideoCapture(os.path.join(root_dir, folder, vid))
                vid_path = os.path.join(root_dir, folder, vid)
                frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                # returns middle or all 60 frames if video frames are more than or equal to 60
                npy_path = os.path.join(root_dir, folder, str(vid_list.index(vid)))
                if frames_length > frames_wanted or frames_length == 60:
                    get_npy_middle(vid_path, 60, npy_path)
                else:
                    pad_frames(vid_path, 60, npy_path)

                print('category: ', folder)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    edit_video_frame_num(60)
