from dataclasses import dataclass
import cv2
import torch
import numpy as np

from typing import Any, Callable, List, Optional
from torch.utils.data import Dataset


@dataclass
class SampledVideoDataset(Dataset):
    video_filenames: List[str]
    num_frames: int
    labels: List[str]
    transform: Optional[Callable] = None

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx):
        video_filename = self.video_filenames[idx]
        label = self.labels[idx]

        # Use OpenCV to load the video
        cap = cv2.VideoCapture(video_filename)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            
        cap.release()

        frames = self.sample_frames(frames, self.num_frames)

        video = np.stack(frames)
        video = torch.from_numpy(video)

        # Apply any transformations
        if self.transform:
            video = self.transform(video)

        return video, label


    @staticmethod
    def sample_frames(frames: List[Any], num_frames: int):
        num_existing_frames = len(frames)
        if num_existing_frames == num_frames:
            return frames
        elif num_existing_frames < num_frames:
            # Loop the frames until we have enough
            return frames * (num_frames // num_existing_frames) + frames[:num_frames % num_existing_frames]
        else:
            # Subsample the frames
            step = num_existing_frames // num_frames
            return frames[::step][:num_frames]
