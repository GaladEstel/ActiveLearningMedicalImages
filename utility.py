from PIL import Image
import numpy as np
import imageio as io

def createAndSaveImage(image_to_save, output_name):
    io.imwrite(output_name, image_to_save)
    return True

def applyMask(input, mask):
    masked = input
    masked[np.where(mask == 0)] = 0
    return masked
