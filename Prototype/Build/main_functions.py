import cv2 as cv
import numpy as np
import face_mesh_connections as fmc

#--- setting up global constants ---
leftEye = {i for pair in fmc.FACEMESH_LEFT_EYE for i in pair}
rightEye = {i for pair in fmc.FACEMESH_RIGHT_EYE for i in pair}
leftIris = {i for pair in fmc.FACEMESH_LEFT_IRIS for i in pair}
rightIris = {i for pair in fmc.FACEMESH_RIGHT_IRIS for i in pair}

def find_available_cameras(option: int = 1, max_index=5):
    if (option == 0):
        return 0
    available_cameras = []
    for i in range(max_index):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return max(available_cameras) if available_cameras else 0

def extract_pupil(w, h, side, faceLandmarks) -> tuple[int, int]:
    x_sum = 0.0
    y_sum = 0.0
    cnt = 0
    for idx, landmark in enumerate(faceLandmarks):
        if (side == 'left' and idx in leftIris) or (side == 'right' and idx in rightIris):
            landmark = faceLandmarks[idx]
            x_sum += int(landmark.x * w)
            y_sum += int(landmark.y * h)
            cnt += 1
    if cnt != 0:
        return (int(x_sum / cnt), int(y_sum / cnt))
    
    return (0, 0)

def horizontal_normalization(pupil, inner, outer, flip_x=False):
    v = inner - outer
    u = pupil - outer
    nx = np.dot(u, v) / np.dot(v, v) #normal value = coefficient of proj(u) on v
    if flip_x is True:
        nx = 1 - nx
    return np.clip(nx, 0.0, 1.0)

def vertical_normalization(eyelid, top_forehead, nose, eyelid_calibration):
    u = eyelid - nose
    v = top_forehead - nose
    u_calibration = eyelid_calibration - nose
    alpha = 2 * np.dot(u_calibration, v) / np.dot(v, v)
    v = v * alpha #alpha = coefficient to make vector (top_forehead - nose) have twice the length as vector (eyelid_calibration - nose) in calibration frame
    ny = np.dot(u, v) / np.dot(v, v) #normal value = coefficient of proj(u) on v
    return ny
