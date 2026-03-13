#---- Work flow ----
# 1. Extract pupil center (x, y) for both eyes
# 2. Extract eye conners (x, y) for both eyes
# 3. Normalize nx:  pupil center to eye conners (horizontal)
# 4. Normalize ny:  pupil center to cheekbone (vertical)
#   4.1. adding calibration key for ny according to frame
#   4.2. making pipeline for screen_projection module part 1 - CURRENT
#   4.3. add EMA filter to smooth out the noise in the output (for both nx and ny)
#   4.4. add Kalman filter to stablize the output
#   4.5. making pipeline for screen_projection module part 2
# 5. project the normalized coordinates to the screen (using a simple linear transformation for now, but can be improved with a more complex model)
# 6. add key for calibration & enabling "INSERT MODE"
# 7. wraping up MVP and start working on Production version
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

#--- video configurations ---
LiveCapture = cv.VideoCapture(mf.find_available_cameras(0))
prevTime = time.time()
frameTimestampMs = 0

glob_v_l_top_eyelid_calibration = (0, 0)
glob_v_r_top_eyelid_calibration = (0, 0)

while True:
    isTrue, frame = LiveCapture.read()
    if isTrue is not True:
        break

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
    frameTimestampMs = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, frameTimestampMs)

    h, w, c = frame.shape
    if results.face_landmarks:
        for faceLandmarks in results.face_landmarks:

            #--- 1. pupil extraction: initialize accumulators ---
            l_pupil_x, l_pupil_y = mf.extract_pupil(w, h, 'left', faceLandmarks)
            r_pupil_x, r_pupil_y = mf.extract_pupil(w, h, 'right', faceLandmarks)

            #--- 2. eye conners: initialize boundary points list ---
            r_outer = faceLandmarks[33]
            r_inner = faceLandmarks[133]
            l_outer = faceLandmarks[263]
            l_inner = faceLandmarks[362]
            
            r_inner_x, r_inner_y = hp.cvt_landmark_to_xy(r_inner, w, h)
            r_outer_x, r_outer_y = hp.cvt_landmark_to_xy(r_outer, w, h)
            l_inner_x, l_inner_y = hp.cvt_landmark_to_xy(l_inner, w, h)
            l_outer_x, l_outer_y = hp.cvt_landmark_to_xy(l_outer, w, h)
            
            #--- 3. normalizing for both pupils to anchor (horizontal) ---
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
            nx = np.clip(nx, 0.0, 1.0)

            #--- 4. y_anchor: initialize vertical boundary points ---
                ##nose + top forehead + both eyelids for vector projection
            nose = faceLandmarks[1]
            nose_x, nose_y = hp.cvt_landmark_to_xy(nose, w, h)
            v_nose = np.array([nose_x, nose_y])

            top_forehead = faceLandmarks[10]
            top_forehead_x, top_forehead_y = hp.cvt_landmark_to_xy(top_forehead, w, h)
            v_top_forehead = np.array([top_forehead_x, top_forehead_y])

            l_top_eyelid = faceLandmarks[159]
            l_top_eyelid_x, l_top_eyelid_y = hp.cvt_landmark_to_xy(l_top_eyelid, w, h)
            v_l_top_eyelid = np.array([l_top_eyelid_x, l_top_eyelid_y])
            
            r_top_eyelid = faceLandmarks[386]
            r_top_eyelid_x, r_top_eyelid_y = hp.cvt_landmark_to_xy(r_top_eyelid, w, h)
            v_r_top_eyelid = np.array([r_top_eyelid_x, r_top_eyelid_y])

                ## normal value calibration
            if cv.waitKey(1) & 0xFF == ord('c'):
                glob_v_l_top_eyelid_calibration = v_l_top_eyelid
                glob_v_r_top_eyelid_calibration = v_r_top_eyelid
                ##left - right & averaging
            l_ny = mf.vertical_normalization(v_l_top_eyelid, v_top_forehead, v_nose, glob_v_l_top_eyelid_calibration)
            r_ny = mf.vertical_normalization(v_r_top_eyelid, v_top_forehead, v_nose, glob_v_r_top_eyelid_calibration)
            ny = (l_ny + r_ny) / 2
            ny = np.clip(ny, 0.0, 1.0)

            print(f"nx: {nx}, ny: {ny}")
            
            ##---=== 4.1 map normal value strictly from [0, 1] ===---
            # Working in screen_projection module

            ##---=== 4.2 pipeline for screen_projection module part 1 ===---
            # Working in screen_projection module


            #--- 5. drawing points for debugging ---
            #hp.check_landmark(frame, faceLandmarks)
            
    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime
    hp.write(frame, "FPS", int(fps), (20, 50), 'green')
    cv.imshow("Face Mesh", frame)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

LiveCapture.release()
cv.destroyAllWindows()