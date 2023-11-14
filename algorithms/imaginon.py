"""
Imaginon class for k-anonymising images ensuring l-diverseness
"""

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from anonymizer import Anonymizer
from cv2 import imshow

# extend Imaginon from Anonymizer, Mondrian-based anonymization
class Imaginon(Anonymizer):
    def __init__(self, df, base_options, options, feature_columns=None, sensitive_column=None):
        super().__init__(df, feature_columns, sensitive_column)
        self.df = df
        self.feature_columns = feature_columns
        self.sensitive_column = sensitive_column
        self.base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        self.options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def anonymise(img, tlx, tly, brx, bry):
        tl = (round(res.face_landmarks[0][tlx].x*img.width), round(res.face_landmarks[0][tly].y*img.height))
        br = (round(res.face_landmarks[0][brx].x*img.width), round(res.face_landmarks[0][bry].y*img.height))

        tmp = cv2.GaussianBlur(img[tl[1]:br[1], tl[0]:br[0]], (69, 69), 0)
        img[tl[1]:br[1], tl[0]:br[0]] = tmp


        res = detector.detect(img)

        cvImage = cv2.cvtColor(img.numpy_view(), cv2.COLOR_RGB2BGR)

        # left eye
        cvImage = blur(cvImage, 471, 470, 469, 472)

        #right eye
        cvImage = blur(cvImage, 476, 475, 474, 477)


        return cvImage
