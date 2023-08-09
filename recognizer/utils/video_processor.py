import os
import random

import cv2
import numpy as np

from multiprocessing import Pool, cpu_count
from typing import List

from recognizer.utils.constants import ROOT_DIR, ALL_DATASET_DIR
from recognizer.utils.utils import get_metadata_from_filename


OUTPUT_DIR = ROOT_DIR / "data" / "all-20percent"

SCALE_FACTOR = 0.2

def get_list_of_videos() -> List[str]:
    file_paths = []

    for file in os.listdir(ALL_DATASET_DIR):
        file_paths.append(str((ALL_DATASET_DIR / file).resolve()))
    
    return file_paths


def process_video(video_file_path: str):

    video_name = video_file_path.split("/")[-1]
    original_video = cv2.VideoCapture(video_file_path)
    

    resized_frames = []

    # Initialize values for new_width and new_height
    new_width = 0
    new_height = 0

    print(f"Starting to process video: {video_name}")

    while(original_video.isOpened()):
        ret, frame = original_video.read()

        if ret:
            height = frame.shape[0]
            width = frame.shape[1]

            new_width = int(width * SCALE_FACTOR)
            new_height = int(new_width * (height / width))

            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            resized_frames.append(resized_frame)

        else:
            break

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    resized_video = cv2.VideoWriter(str(OUTPUT_DIR / video_name), fourcc, 60.0, (new_width, new_height))

    for frame in resized_frames:
        resized_video.write(frame)

    original_video.release()
    resized_video.release()


if __name__ == "__main__":
    # Using half of the available CPUs for parallel processing (you can adjust this number).
    num_processes = cpu_count()

    file_paths = get_list_of_videos()

    print(f"About to resize {len(file_paths)} videos in {num_processes} processes...")

    with Pool(num_processes) as pool:
        pool.map(process_video, file_paths)
