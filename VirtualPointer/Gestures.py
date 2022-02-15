import mouse
import math
import time


class GestureProcessor:
    def __init__(self, tracker):
        self.tracker = tracker
        self.flag_dict = dict()
        self.eye_lms = None
        self.left_hand_lms = None
        self.right_hand_lms = None
        self.mouse_hold = False
        for i in range(5):
            self.flag_dict[i + 1] = False

    def __process_operations(self):
        if self.flag_dict[3] is True:
            return
        if self.flag_dict[1] is True and self.mouse_hold is False:
            if self.flag_dict[5] is False:
                mouse.double_click('left')
            else:
                mouse.click('left')
        if self.flag_dict[2] is True and self.mouse_hold is False:
            mouse.click('right')
        if self.flag_dict[5] is True:
            if self.flag_dict[4] is True:
                mouse.press('left')
                self.mouse_hold = True
            else:
                if self.mouse_hold is True:
                    mouse.release('left')
                    self.mouse_hold = False
            cx, cy = self.right_hand_lms[0]
            mouse.move(cx * 2 - 1922, cy * 2 - 1080)

    def process_gestures(self):
        self.eye_lms = self.tracker.get_eye_landmarks()
        self.left_hand_lms = self.tracker.get_left_hand_landmarks()
        self.right_hand_lms = self.tracker.get_right_hand_landmarks()
        if self.eye_lms is not None:
            self.flag_dict[1] = self.__left_eye_blink()
            self.flag_dict[2] = self.__right_eye_blink()
            self.flag_dict[3] = self.flag_dict[1] and self.flag_dict[2]
        if self.left_hand_lms is not None:
            self.flag_dict[4] = self.__hold()
        if self.right_hand_lms is not None:
            self.flag_dict[5] = self.__move_mode()
        self.__process_operations()
        for i in range(5):
            self.flag_dict[i + 1] = False

    def __left_eye_blink(self):
        if math.dist(self.eye_lms[0], self.eye_lms[1]) < 18.0:
            time.sleep(0.5)
            return True
        else:
            return False

    def __right_eye_blink(self):
        if math.dist(self.eye_lms[2], self.eye_lms[3]) < 18.0:
            time.sleep(0.5)
            return True
        else:
            return False

    def __hold(self):
        tip_lm = self.left_hand_lms[0]
        thumb_lm = self.left_hand_lms[4]
        if math.dist(tip_lm, thumb_lm) < 80.0:
            return True
        else:
            return False

    def __move_mode(self):
        tip_lm = self.right_hand_lms[0]
        thumb_lm = self.right_hand_lms[4]
        if thumb_lm[1] > tip_lm[1]:
            return True
        else:
            return False
