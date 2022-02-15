import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import time
import cv2


class CustomHolisticTracker:
    def __init__(self, mode=False, refine=True, detection_conf=0.6, tracking_conf=0.6, complexity=0):
        self.mode = mode
        self.refine = refine
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf
        self.complexity = complexity
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(self.mode, self.complexity, True, False, True, self.refine,
                                                  self.detection_conf, self.tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils
        self.res = None
        self.landmark_face_subset = None
        self.landmark_right_hand_subset = None
        self.landmark_left_hand_subset = None
        self.h = None
        self.w = None
        self.c = None
        self.eye_lm_list = list()
        self.right_lm_list = list()
        self.left_lm_list = list()

    def process_frame(self, img):
        self.h, self.w, self.c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.flip(img_rgb, 1)
        self.res = self.holistic.process(img_rgb)
        img = cv2.flip(img, 1)
        return img

    def get_eye_landmarks(self):
        flag = True
        if self.res.face_landmarks is not None:
            flag = False
            self.landmark_face_subset = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    self.res.face_landmarks.landmark[159],
                    self.res.face_landmarks.landmark[145],
                    self.res.face_landmarks.landmark[386],
                    self.res.face_landmarks.landmark[374],
                ])
            self.eye_lm_list.clear()
            for i in range(4):
                x, y = self.landmark_face_subset.landmark[i].x, self.landmark_face_subset.landmark[i].y
                cx, cy = int(x * self.w), int(y * self.h)
                self.eye_lm_list.append((cx, cy))

        if flag is True:
            time.sleep(0.5)

        if len(self.eye_lm_list) > 0:
            return self.eye_lm_list

    def draw_landmarks(self, img, eye=True, left=True, right=True):
        eye_lm_list = self.get_eye_landmarks()
        left_lm_list = self.get_left_hand_landmarks()
        right_lm_list = self.get_right_hand_landmarks()
        if eye_lm_list is not None and eye is True:
            self.mp_draw.draw_landmarks(img, self.landmark_face_subset)
        if left_lm_list is not None and left is True:
            self.mp_draw.draw_landmarks(img, self.landmark_left_hand_subset)
        if right_lm_list is not None and right is True:
            self.mp_draw.draw_landmarks(img, self.landmark_right_hand_subset)

    def get_right_hand_landmarks(self):
        if self.res.left_hand_landmarks is not None:
            self.landmark_right_hand_subset = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    self.res.left_hand_landmarks.landmark[8],
                    self.res.left_hand_landmarks.landmark[12],
                    self.res.left_hand_landmarks.landmark[16],
                    self.res.left_hand_landmarks.landmark[20],
                    self.res.left_hand_landmarks.landmark[4]
                ])

            self.right_lm_list.clear()
            for i in range(5):
                x, y = self.landmark_right_hand_subset.landmark[i].x, self.landmark_right_hand_subset.landmark[i].y
                cx, cy = int(x * self.w), int(y * self.h)
                self.right_lm_list.append((cx, cy))
            return self.right_lm_list

    def get_left_hand_landmarks(self):
        if self.res.right_hand_landmarks is not None:
            self.landmark_left_hand_subset = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    self.res.right_hand_landmarks.landmark[8],
                    self.res.right_hand_landmarks.landmark[12],
                    self.res.right_hand_landmarks.landmark[16],
                    self.res.right_hand_landmarks.landmark[20],
                    self.res.right_hand_landmarks.landmark[4]
                ])

            self.left_lm_list.clear()
            for i in range(5):
                x, y = self.landmark_left_hand_subset.landmark[i].x, self.landmark_left_hand_subset.landmark[i].y
                cx, cy = int(x * self.w), int(y * self.h)
                self.left_lm_list.append((cx, cy))
            return self.left_lm_list
