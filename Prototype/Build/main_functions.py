import cv2 as cv
import numpy as np

def horizontal_normalization(pupil, inner, outer, flip_x=False):
    v = inner - outer
    u = pupil - outer
    nx = np.dot(u, v) / np.dot(v, v) #normal value = coefficient of proj(u) on v
    if flip_x is True:
        nx = 1 - nx
    return np.clip(nx, 0.0, 1.0)

def vertical_normalization(pupil, chin, coefficent):
    u = pupil - chin
    u_y = u[1]
    v = coefficent
    ny = u_y / v #normal value = coefficient of proj(u) on v
    return ny