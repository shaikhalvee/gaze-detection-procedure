import cv2
import numpy as np
import dlib


def gaze_detect():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        for face in faces:
            # print(face)
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            w, h = face.width(), face.height()
            # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            x = landmarks.part(36).x
            y = landmarks.part(36).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gaze_detect()
