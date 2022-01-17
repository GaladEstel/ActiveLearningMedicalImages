import numpy as np
import matplotlib.pyplot as plt
import cv2

def reconstruct(clustered_images, set, set_type, dataset_name):

    if dataset_name == "STARE" and set_type == "train":   # first (original) dataset
        # create a numpy array which will contain ordered patches
        # since we discarded some black patches we initialize this array with full zeros (black)
        # originally we had 18*18 = 324 patches (32x32 size) per image, and we have 16 images in train so:
        shape = (5184, 32, 32)  # gray images
        tot_images = 16
        ordered_patches = np.zeros(shape)
        # we iterate over the patches name and we put the correspondant clustered image in the array of ordered patches
        for i,file_name in enumerate(set):
            # pick the image id and the position
            data = []  # x,y,id
            file_name_extract = file_name.replace("_", " ")
            for word in file_name_extract.split():
                if word.isdigit(): data.append(int(word))
            ordered_patches[324 * (data[2]-21) + data[0]*18 + data[1]] = clustered_images[i]  # cv2.cvtColor(clustered_images[i], cv2.COLOR_RGB2GRAY)

        final_images = np.zeros((16,576,576))  # 16 images 576x576
        for iter in range(16):
            for i in range(18): # along the rows
                for j in range(18):  # along the columns
                    if j == 0:
                        toAttachH = ordered_patches[iter*324 + i*18 + j]
                    else:
                        toAttachH = np.hstack((toAttachH, ordered_patches[iter*324 + i*18 + j]))
                if i == 0:
                    toAttachV = toAttachH
                else:
                    toAttachV = np.vstack((toAttachV, toAttachH))
            final_images[iter] = toAttachV


        # for image in final_images:
        #     plt.imshow(image,"gray")
        #     plt.show()
        return final_images