# used link:
# https://towardsdatascience.com/cnn-based-face-detector-from-dlib-c3696195e01c

# import required packages
import cv2
import dlib
import argparse
import time

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
args = ap.parse_args()

# load input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# applying HOG face detection
start = time.time()

# apply face detection (hog)
faces_hog = hog_face_detector(image, 1)

end = time.time()
print("Execution time (in seconds): ")
print("HOG: ", format(end - start, '.2f'))

# loop over detected faces
for face in faces_hog:
    x = face.left()
    y = face.top() - 20
    w = face.right() - x
    h = face.bottom() - y

    # draw box over face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# write at the top left corner of the image
# for color identification
img_height, img_width = image.shape[:2]
cv2.putText(image, "HOG", (img_width - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# display output image
cv2.imshow("face detection with dlib", image)
cv2.waitKey()

# save output image
cv2.imwrite("face_detection.png", image)

# close all windows
cv2.destroyAllWindows()
