from cv2 import cv2
import os
import numpy as np


class DomainRandomizer(object):
    def __init__(self, data_folder):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        self.masks = []
        for image_name in os.listdir(os.path.join(package_directory, data_folder)):
            image = cv2.imread(os.path.join(package_directory, data_folder, image_name))
            image = np.transpose(image, (2, 0, 1))[0]
            mask = np.sqrt(image.astype(np.float) / np.max(image))
            self.masks.append(mask)

    def get_mask(self):
        rnd = np.random.randint(0, len(self.masks))
        img = self.masks[rnd]
        return img
        (h, w) = img.shape
        center = (w / 2, h / 2)
        angle = np.random.randint(0, 360)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(img, matrix, (h, w))
        return rotated
