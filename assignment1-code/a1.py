import sys
import math
import cv2
from cv2 import threshold
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.ndimage.filters import maximum_filter1d




# main function
def main():
    # if the function call is incorrect
    if len(sys.argv) != 2:
            print("Incorrect number of paramaters!!\nPlease run this program using the following:\n'python a1.py <input_image>'")
    
    # if the function call is correct, run the program
    else:
        # read grayscale image & initialize
        image = cv2.imread(sys.argv[1], 0)
        h, w = image.shape

        # display original image image
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)

        plt.hist(image.ravel(),256,[0,256]); plt.show()

        threshold = input("Threshold?   ")
        print("Binarizing image with threshold " + threshold + "...")

        # binarize image based on threshold
        for i in range(0, h):
            for j in range(0, w):
                if image[i][j] < int(threshold):
                    image[i][j] = 0
                else:
                    image[i][j] = 255



        # display binarized image image
        cv2.imshow('Binarized Image', image)
        cv2.waitKey(0)





main()