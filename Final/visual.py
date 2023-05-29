import cv2
import numpy as np
from glob import glob
import csv
import matplotlib.pyplot as plt


path = './seq1/dataset/'
folders = sorted(glob(path + '*'))

first_folder = folders[1]

image = cv2.imread(first_folder + '/raw_image.jpg').astype(np.uint8)

coord = []  # (x1, y1, x2, y2, class_id, probability)
with open(first_folder + '/detect_road_marker.csv', 'r') as f:
    reader = csv.reader(f)
    for l in reader:
        coord.append(l)

for x1, y1, x2, y2, class_id, probability in coord:
    image = cv2.circle(image, (int(float(x1)), int(float(y1))), radius=10, color=(0, 255, 0), thickness=-1)
    image = cv2.circle(image, (int(float(x2)), int(float(y2))), radius=10, color=(0, 255, 0), thickness=-1)

plt.imshow(image)
plt.show()
pass
