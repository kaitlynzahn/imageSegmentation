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


        # initialize k indexes for random centers
        randomCenters = np.array([])
        for x in range(k):
                val = random.randint(0, len(pixel_vals))
                randomCenters = np.append(randomCenters, val)



        # iterate __ times
        for i in range(1):
                # go through every pixel in the image
                for j in range(0, len(pixel_vals)):
                        print(j)
                        dist = 1000
                        chosenCenter = randomCenters[0]
                        # for each pixel, find the distance from each center
                        for l in range(0, k):
                                # choose the center with the minimum distance
                                if distance.euclidean(pixel_vals[j], pixel_vals[int(randomCenters[l])]) < dist:
                                        dist = distance.euclidean(pixel_vals[j], pixel_vals[int(randomCenters[l])])
                                        chosenCenter = randomCenters[l]
                        # assign the pixel to the color of the center it's closest to
                        pixel_vals[j] = pixel_vals[int(chosenCenter)]
                # move the center to the average of the pixels

        # convert data into 8-bit values
        pixel_vals = np.uint8(pixel_vals)
        # segmented_data = pixel_vals[labels.flatten()]

        # reshape data into the original image dimensions
        segmented_image = pixel_vals.reshape((image.shape))

        # connected component labelling here I think
        # cv2.connectedComponents(segmented_image)

        # display segmented image image
        cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)

main()