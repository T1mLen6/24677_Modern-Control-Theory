import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.linalg import svd
import cv2

def compress_svd(image, K):
    U,s,V = svd(image,full_matrices=False)
    reconst_M = np.dot(U[:,:K], np.dot(np.diag(s[:K]), V[:K,:]))
    
    return reconst_M

def compress_gray_img(image, ratio):
    m = image.shape[0]
    n = image.shape[1]
    K = (ratio/100.0) * ((m*n)/(1+m+n))
    reconst_image = compress_svd(image, K)
    return reconst_image

img = cv2.imread('CMU_Grayscale.png', 0) 
ratio = 50
final_img = compress_gray_img(img, ratio)

# Display the image
cv2.imshow("Image", final_img)
 
# Wait for the user to press a key
cv2.waitKey(0)
 
# Close all windows
cv2.destroyAllWindows()