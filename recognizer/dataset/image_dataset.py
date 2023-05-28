from dataclasses import dataclass
import cv2
import torch
import numpy as np

from typing import Any, Callable, List, Optional
from torch.utils.data import Dataset
from torchvision import transforms

@dataclass
class ImageDataset(Dataset):
    video_filenames: List[str]
    labels: List[int]
    transform: Optional[Callable] = None

    def __len__(self):
        return len(self.video_filenames)

    def __getitem__(self, idx: int):
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

        frame = frames[-1]

        image = torch.from_numpy(np.array(frame))

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, label
