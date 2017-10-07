# This script shows how to get a heatmap of a CNN feature
# Author: yongyuan.name

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('./data/paris.png')

data = cv2.imread('./data/spatial_weights.jpg', 1)
gray_image = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots(1)
implot = ax.imshow(img)
heatmap = ax.pcolor(gray_image, alpha=0.6)
plt.axis('off')
plt.savefig("./data/test.png")

