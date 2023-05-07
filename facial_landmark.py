"""
For finding the face and face landmarks for further manipulication
"""

import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Facemesh
        '''
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )
        
        '''

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        filter_imageBGRA = cv2.imread('media/filter.png', cv2.IMREAD_UNCHANGED)
        filter_on = False
        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        frame_height, frame_width, _ = img.shape

        # Resize the filter image to the size of the frame.
        filter_imageBGRA = cv2.resize(filter_imageBGRA, (frame_width, frame_height))

        # Get the three-channel (BGR) image version of the filter image.
        filter_imageBGR = filter_imageBGRA[:, :, :-1]

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.holistic.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.faces = []

        if self.results.face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=self.results.face_landmarks,
                connections=self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1))

            face = []
            for id, lmk in enumerate(self.results.face_landmarks.landmark):
                x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                face.append([x, y])

                # show the id of each point on the image
                # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            self.faces.append(face)

        if self.results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=self.results.left_hand_landmarks,
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2))

        if self.results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=self.results.right_hand_landmarks,
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2))

            if self.results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < \
                    self.results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y:
                print('Index finger pointing up')
                filter_on = True

            if self.results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > \
                    self.results.right_hand_landmarks.landmark[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y:
                print('middle finger pointing up')
                filter_on = False

        if filter_on:
            img[filter_imageBGRA[:, :, -1] == 255] = filter_imageBGR[filter_imageBGRA[:, :, -1] == 255]

        return img, self.faces
'''
        if self.results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=img,
                landmark_list=self.results.pose_landmarks,
                connections=self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                             circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2))      '''


# sample run of the module
def main():

    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)

        #if faces:
         #   print(faces[0])

        cv2.imshow('MediaPipe FaceMesh', img)

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    # demo code
    main()
