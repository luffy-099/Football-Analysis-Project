import cv2
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
from numpy.linalg import norm


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames =[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def cosine_similarity(a, b):
    if norm(a) == 0 or norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (norm(a) * norm(b))

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final classifier
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, frame, bbox):
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            # Return a zero vector if crop is invalid
            return np.zeros(512)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(crop).squeeze().cpu().numpy()
        return feat  # shape: (512,)