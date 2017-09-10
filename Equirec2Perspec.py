import os
import sys
import cv2
import numpy as np

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS = 128):
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
        ''' 
        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        #interval = w_len / (width - 1)
        interval = 1
        x_range = (np.arange(0, width) - c_x) * interval
        x_range = np.arctan(x_range / RADIUS) / np.pi * 180

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        #interval = h_len / (height - 1)
        interval = 1
        y_range = (np.arange(0, height) - c_y) * interval
        y_range = np.arctan(y_range / RADIUS) / np.pi * 180
        #print x_range
        #print y_range[0], y_range[-1]
        #exit()
        x_range += THETA
        y_range -= PHI

        x_range = x_range / 180 * equ_cx + equ_cx
        y_range = y_range / 90 * equ_cy + equ_cy

        x_grid = np.tile(x_range, [height, 1])
        y_grid = np.tile(y_range, [width, 1]).T

        for x in range(width):
            for y in range(height):
                cv2.circle(self._img, (int(x_grid[y, x]), int(y_grid[y, x])), 1, (0, 255, 0))
        return self._img 
        '''
        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], np.float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]
        
        [R, _] = cv2.Rodrigues(np.array([0, 1, 0], np.float32) * np.radians(-PHI))
        #[R, _] = cv2.Rodrigues(np.array([0, 0, 1], np.float32) * np.radians(THETA))
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R, xyz).T
        #print xyz.reshape([height, width, 3])[360, 540, 0]
        #print y_map[0, 0]
        #print z_map[0, 0]
        #print xyz
        #np.save('/Users/fu-en.wang/SandBox/pygame/t.npy', xyz)
        #exit()
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)
        
        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        #print lon[0, :200]
        #print theta.reshape([height, width])[0, :200]
        #print lat
        #print lon
        #print lat
        #exit()
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        #for x in range(width):
        #    for y in range(height):
        #        cv2.circle(self._img, (int(lon[y, x]), int(lat[y, x])), 1, (0, 255, 0))
        #return self._img 
    
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        #persp = cv2.remap(self._img, x_grid, y_grid, cv2.INTER_LINEAR)
        return persp
        






