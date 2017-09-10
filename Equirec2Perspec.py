import os
import sys
import cv2
import numpy as np

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        x_range = (np.arange(0, width) - c_x) / c_x * wFOV / 2
        x_range += THETA
        #print x_range
        y_range = (np.arange(0, height) - c_y) / c_y * hFOV / 2
        y_range -= PHI
        
        idx1 = y_range > 90
        idx2 = y_range < -90
        y_range[idx1] = 90 - (y_range[idx1] - 90)
        y_range[idx2] = -90 - (y_range[idx2] + 90)
        
        #x_range = x_range / 180 * equ_cx + equ_cx
        x_range = x_range
        y_range = y_range / 90 * equ_cy + equ_cy
        
        x_grid = np.tile(x_range, [height, 1]).astype(np.float32)
        y_grid = np.tile(y_range, [width, 1]).T.astype(np.float32)

        x_grid[idx1, :] -= 180

        x_grid[idx2, :] -= 180
        #x_grid[idx2, :][x_grid[idx2, :] < 0] += 180
        x_grid = x_grid / 180 * equ_cx + equ_cx
        #print x_grid
        #print y_grid

        persp = cv2.remap(self._img, x_grid, y_grid, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        #persp = cv2.remap(self._img, x_grid, y_grid, cv2.INTER_LINEAR)
        return persp
        






