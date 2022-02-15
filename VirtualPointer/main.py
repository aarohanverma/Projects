from HolisticTrackingModule import *
from Gestures import *

cap = cv2.VideoCapture(0)
tracker = CustomHolisticTracker()
gs = GestureProcessor(tracker)
while True:
    check, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    if check is True:
        frame = tracker.process_frame(frame)
        gs.process_gestures()
        tracker.draw_landmarks(frame)
        #cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
