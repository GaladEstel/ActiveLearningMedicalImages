import numpy as np
import cv2
import matplotlib.pyplot as plt

def kmeans(patch_image, prediction):

    # map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'

    norm_image = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # norm_image = patch_image
    norm_image = norm_image.astype(np.uint8)
    # norm_image = cv2.cvtColor(norm_image,cv2.COLOR_RGB2GRAY)
    # norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)
    # vectorized = np.float32(norm_image.reshape((-1, 1)))
    vectorized = np.float32(norm_image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10

    # value that we want to add to the classes in an image (to be more precise avoiding noise)
    addition = 1
    # check if there is black in the patch (i.e. if the patch represents a piece of border of the eye)
    # a black pixel is close to [0,0,0] RGB -> the mean will be close to 0
    avg_rgb_image = np.mean(norm_image,axis=2)
    # if 0 in norm_image:
    if np.any(avg_rgb_image <= 5):
        if prediction == 'non-vessel':  # black and eye without vessels
            K = 2
        else:  # black, eye vessels and rest of the eye
            K = 3 + addition + 1
    else:
        if prediction == 'non-vessel':  # eye without vessels
            K = 1
        else:  # eye vessels and rest of the eye
            K = 2 + addition


    _, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # Assign to each pixel its centroid
    res = center[label.flatten()]
    result_image = res.reshape(norm_image.shape)

    # Plot
    figure_size = 15
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(norm_image)#, 'gray')
    plt.title('Original Image (normalized)')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)#, 'gray')
    plt.title(f'Segmented Image when K = {K} - {prediction}')
    plt.xticks([])
    plt.yticks([])
    plt.show()

