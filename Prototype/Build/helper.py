import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=0.15):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    #live video
    #capture.set(3,width)
    #capture.set(4,height)
    return 0

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

def draw(frame, tuple, color):
    x, y = tuple
    color_f = (255, 255, 255)
    if color == 'red':
        color_f = (0, 0, 255)
    elif color == 'green':
        color_f = (0, 255, 0)
    elif color == 'blue':
        color_f = (255, 0, 0)
    elif color == 'yellow':
        color_f = (0, 255, 255)
    elif color == 'purple':
        color_f = (255, 0, 255)
    cv.circle(frame, (x, y), 1, color_f, 1)

def write(frame, text, variable, position, color):
    if (color == 'red'):
        color = (0, 0, 255)
    elif (color == 'green'):
        color = (0, 255, 0)
    elif (color == 'blue'):
        color = (255, 0, 0)
    elif (color == 'yellow'):
        color = (0, 255, 255)
    elif (color == 'purple'):
        color = (255, 0, 255)
    cv.putText(frame, f"{text}: {float(variable)}", position, cv.FONT_HERSHEY_COMPLEX, 1, color, 2)

def cvt_landmark_to_xy(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def check_landmark(frame, faceLandmarks):
    h, w, _ = frame.shape
    for idx, landmark in enumerate(faceLandmarks):
        x, y = cvt_landmark_to_xy(landmark, w, h)
        draw(frame, (x, y), 'green')
