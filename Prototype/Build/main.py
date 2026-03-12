#---- Work flow ----
# 1. Extract pupil center (x, y) for both eyes
# 2. Extract eye conners (x, y) for both eyes
# 3. Normalize nx:  pupil center to eye conners (horizontal)
# 4. Normalize ny:  pupil center to cheekbone (vertical) - CURRENT
#   4.1. adding calibration key for ny according to frame
# 5. add EMA filter to smooth out the noise
# 6. add Kalman filter to stablize the output
# 7. project the normalized coordinates to the screen (using a simple linear transformation for now, but can be improved with a more complex model)
# 8. add key for calibration & enabling "INSERT MODE"
# 9. wraping up MVP and start working on Production version
#--- End of work flow ---

import helper as hp
import face_mesh_connections as fmc
import main_functions as mf
import cv2 as cv
import numpy as np
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.VIDEO,
                                       num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

print("Landmarker created successfully")
#End defining

#--- setting up global constants ---
leftEye = {i for pair in fmc.FACEMESH_LEFT_EYE for i in pair}
rightEye = {i for pair in fmc.FACEMESH_RIGHT_EYE for i in pair}
leftIris = {i for pair in fmc.FACEMESH_LEFT_IRIS for i in pair}
rightIris = {i for pair in fmc.FACEMESH_RIGHT_IRIS for i in pair}
min_x, min_y = 10**9, 10**9
max_x, max_y = -1, -1

#--- video figurations ---
LiveCapture = cv.VideoCapture(0)
prevTime = time.time()
frameTimestampMs = 0

def extract_pupil(w, h, side, faceLandmarks) -> (int, int):
    x_sum = 0.0
    y_sum = 0.0
    cnt = 0
    for idx, landmark in enumerate(faceLandmarks):
        if (side == 'left' and idx in leftIris) or (side == 'right' and idx in rightIris):
            landmark = faceLandmarks[idx]
            h, w, _ = frame.shape
            x_sum += int(landmark.x * w)
            y_sum += int(landmark.y * h)
            cnt += 1
    if cnt != 0:
        return (int(x_sum / cnt), int(y_sum / cnt))
    
    return (0, 0)

def cvt_landmark_to_xy(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

while True:
    isTrue, frame = LiveCapture.read()
    if isTrue is not True:
        break

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
    frameTimestampMs = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, frameTimestampMs)

    if results.face_landmarks:
        for faceLandmarks in results.face_landmarks:
            h, w, _ = frame.shape

            #--- pupil extraction: initialize accumulators ---
            l_pupil_x, l_pupil_y = extract_pupil(w, h, 'left', faceLandmarks)
            r_pupil_x, r_pupil_y = extract_pupil(w, h, 'right', faceLandmarks)

            #--- eye conners: initialize boundary points list ---
            r_outer = faceLandmarks[33]
            r_inner = faceLandmarks[133]
            l_outer = faceLandmarks[263]
            l_inner = faceLandmarks[362]
            
            r_inner_x, r_inner_y = cvt_landmark_to_xy(r_inner, w, h)
            r_outer_x, r_outer_y = cvt_landmark_to_xy(r_outer, w, h)
            l_inner_x, l_inner_y = cvt_landmark_to_xy(l_inner, w, h)
            l_outer_x, l_outer_y = cvt_landmark_to_xy(l_outer, w, h)

            #--- normalizing for both pupils to anchor (horizontal) ---
                ## right normalization
            r_outer = np.array([r_outer_x, r_outer_y])
            r_inner = np.array([r_inner_x, r_inner_y])
            r_pupil = np.array([r_pupil_x, r_pupil_y])
            r_nx = mf.horizontal_normalization(r_pupil, r_inner, r_outer, False)
                ## left normalization
            l_outer = np.array([l_outer_x, l_outer_y])
            l_inner = np.array([l_inner_x, l_inner_y])
            l_pupil = np.array([l_pupil_x, l_pupil_y])
            l_nx = mf.horizontal_normalization(l_pupil, l_inner, l_outer, True)
                ## averaging 
            nx = (l_nx + r_nx) / 2

            #--- y_anchor: initialize vertical boundary points ---
            nose = faceLandmarks[1]
            nose_x, nose_y = cvt_landmark_to_xy(nose, w, h)
            ## right normalization
            r_ny = r_pupil_y - nose_y
            ## left normalization
            l_ny = l_pupil_y - nose_y
            ## averaging
            ny = (l_ny + r_ny) / 2
            print(l_ny)

            #--- drawing points for debugging ---
            hp.draw(frame, (r_inner_x, r_inner_y), 'red')
            hp.draw(frame, (r_outer_x, r_outer_y), 'red')
            hp.draw(frame, (l_inner_x, l_inner_y), 'red')
            hp.draw(frame, (l_outer_x, l_outer_y), 'red')
            hp.draw(frame, (int(l_pupil_x), int(l_pupil_y)), 'white')
            hp.draw(frame, (int(r_pupil_x), int(r_pupil_y)), 'white')
            cv.putText(frame, f"ny: {int(ny)}",(20, 40),cv.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 2)

    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime


    cv.imshow("Face Mesh", frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

LiveCapture.release()
cv.destroyAllWindows()