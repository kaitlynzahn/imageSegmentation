from cProfile import label
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from scipy.spatial import distance


# main function
def main():
    # if the function call is incorrect
    if len(sys.argv) != 2:
            print("Incorrect number of paramaters!!\nPlease run this program using the following:\n'python a1b.py <input_image>'")
    
    # if the function call is correct, run the program
    else:
        # read rgb image & initialize
        image = cv2.imread(sys.argv[1], 1)

        # Change color to RGB (from BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # display original image image
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)

        # getting k value from user
        k = int(input("Select a k value: "))
        print("Segmenting using kmeans...")

        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals = image.reshape((-1,3))
        # Convert to float 
        pixel_vals = np.float32(pixel_vals)

        # set criteria to stop running after 100 iterations or the required accuracy becomes 85%
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        # random centers chosen
        # k-means clustering with random centers
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        # reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))

        # display segmented image image
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)


main()