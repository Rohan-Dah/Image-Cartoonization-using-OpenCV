import numpy as np
import cv2 as cv

#initiating cam
cap = cv.VideoCapture(0)

#infinite loop to capture the frames
while(True):
    ret, img = cap.read()

    #reshaping the image to be fed into the kmeans data
    z = img.reshape((-1, 3))

    #Converting it to float
    z = np.float32(z)

    #define criteria, number of clusters(k) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 3, 0.9)

    #Number of Clusters to be defined
    k = 9
    ret, label, center = cv.kmeans(z, k, None, criteria, 1, cv.KMEANS_RANDOM_CENTERS)

    #Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    #increasing the intensity for brighter image
    res2 = res2 + 15

    cv.imshow("Segmented", res2)
    #Exiting the apploication om pressing q

    if(cv.waitKey(10)==ord('q')):
        break

cap.release()
cv.destroyAllWindows()    
