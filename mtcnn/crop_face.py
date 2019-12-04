
from src.data.preprocess import image_preprocess
from test.mtcnn.mtcnn_detector import get_mtcnn_detector
import numpy as np
import cv2
import os
from tqdm import tqdm

face_path = '../../data/CK+/SCK+/val/6'


class CropFace:
    def __init__(self):
        self.detector = get_mtcnn_detector(prefix=['./MTCNN_model/PNet_landmark/PNet', './MTCNN_model/RNet_landmark/RNet', './MTCNN_model/ONet_landmark/ONet'])

    def crop_face(self):
        imgs = os.listdir(face_path)
        for img_path in tqdm(imgs):
            img = cv2.imread(os.path.join(face_path, img_path), 0)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            faces, landmarks = self.detector.detect(img_rgb)
            for face in faces:
                x1, y1, x2, y2 = face[:4]
                x1 = max(0, int(x1) - 5)
                y1 = max(0, int(y1) - 5)
                x2 = min(490, int(x2) + 5)
                y2 = min(640, int(y2) + 5)
                face_img = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(face_path, img_path), face_img)


if __name__ == '__main__':
    ct = CropFace()
    ct.crop_face()
