#--- Goal | Work flow ---
# Summary: This file is responsible for projecting the normalized coordinates to the screen.
# We need from input - or assumption about:
# - dist_screen_to_face
# - screen_width, screen_height
# - theta (bounding left/right eye angle to the calibrated center), phi (bounding up/down eye angle to the calibrated center) [these bounded the 0-1 of sx, sy]

# The main pipeline is as follows:
# I. 1/ fresh nx, ny -> 2/ clean nx, ny (looking at left edge = 0, right edge = 1, top edge = 0, bottom edge = 1) -> 3/ project to screen coordinates (sx, sy) -> 4/ move mouse to (sx, sy)
# ~1/ Take from main.py
# ~2/ Calculate dist_screen_to_eyes using focal length -> getting theta(phi) using tan(screen_width(height)/dist_screen_to_eyes)
# ~2.1/ angle(nx, center) = coeff1 * theta, angle(ny, center) = coeff2 * phi, make sure they stay under 100% (when trying to look outside the bounds)
# ~2.2/ using coeff1, coeff2, these are the calibrated (0-1) nx, ny -> use it for (nx, ny) -> (sx, sy) translation
# ~3/ using sx, sy, screen_width, screen_height to calculate the projected (nx, ny) on the screen -> use pyautogui to move mouse to (sx, sy)
# ~4/ Add EMA filter to smooth out the noise in the output (for both sx and sy)

# During calibration
