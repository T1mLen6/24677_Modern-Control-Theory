import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.linalg import svd
import cv2
from PIL import Image


# def compress_svd(image, K):
#     U,s,V = svd(image)
#     reconst_M = np.dot(np.dot(U[:,:K], np.diag(s[:K])), V[:K,:])
    
#     return reconst_M, s

def compress_gray_img(image, ratio):
    m = image.shape[0]
    n = image.shape[1]
    K = round((ratio/100) * ((m*n)/(1+m+n)))
    print(K,m,n)
    U,s,V = svd(image)
    reconst_M = np.dot(np.dot(U[:,:K], np.diag(s[:K])), V[:K,:])
    sing_val = np.linalg.matrix_rank(reconst_M)
    print("The number of singular values is: ", sing_val)
    return reconst_M, sing_val

img = cv2.imread('CMU_Grayscale.png', 0) 
#img = Image.open('CMU_Grayscale.png')
#img = img.convert('LA')
ratio = 50
final_img, sing_val = compress_gray_img(img, ratio)
#plt.imshow(final_img, cmap='gray')


for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if final_img[i,j] <= 0:
            final_img[i,j] = 0
            
        if final_img[i,j] >= 255:
            final_img[i,j] = 255
            
            
    
# Display the image
final_img = final_img.astype(np.uint8)
cv2.imshow("Image", final_img)
cv2.imwrite(str(ratio) + '%' + ' compression ratio_sing_val_' + str(sing_val) + '.png', final_img)
#plt.imshow(final_img, plt.cm.gray_r)
 
while True:
    # Check if any key is pressed
    key = cv2.waitKey(1) & 0xFF
    
    # If the "q" key is pressed or the window is closed, break the loop
    if key == ord("q") or key == 27:  # "q" or Esc key
        break


cv2.waitKey()

# close all open windows
cv2.destroyAllWindows()
