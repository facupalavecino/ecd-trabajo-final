from dataclasses import dataclass
import cv2
import torch
import numpy as np

from typing import Callable, List, Optional
from torch.utils.data import Dataset


@dataclass
class VideoDataset(Dataset):
    video_filenames: List[str]
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

        # Stack frames into a tensor
        video = np.stack(frames)
        video = torch.from_numpy(video)

        # Apply any transformations
        if self.transform:
            video = self.transform(video)

        return video, label
