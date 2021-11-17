from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
import math

facial_features_cordinates = {}
faces_array =  []
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-i2", "--image2", required=True,
	help="path to input image")
ap.add_argument("-i3", "--image3", required=True,
	help="path to input image")
args = vars(ap.parse_args())


def shape_to_numpy_array(shape, dtype="int"):

    coordinates = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=1):

    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [(0,0,0), (0,0,0), (0,0,0),
                  (0,0,0), (0,0,0),
                  (0,0,0), (0,0,0)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        print("\n\nhello")
        print(i,name,"\n")
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        if name == "Jaw":
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    print(facial_features_cordinates)
    return output

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])
image = cv2.resize(image, (300,300) , interpolation =  cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier('frontface.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

image2 = cv2.imread(args["image2"])
image2 = cv2.resize(image2, (300,300) , interpolation =  cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier('frontface.xml')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

image3 = cv2.imread(args["image3"])
image3= cv2.resize(image3, (300,300) , interpolation =  cv2.INTER_AREA)
face_cascade = cv2.CascadeClassifier('frontface.xml')
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
faces3 = face_cascade.detectMultiScale(gray3, 1.1, 4)



faces2 = faces2
faces3 = faces3
pointsf1 = []
pointsf2 =[]
pointsf3 = []
print(faces_array)
for (x, y, w, h) in faces:
    for i in range(x,x+w):
        for j in range(y,y+h):
            pointsf1.append([i,j])
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces2:
    for i in range(x,x+w):
        for j in range(y,y+h):
            pointsf2.append([i,j])
    cv2.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 2)
for (x, y, w, h) in faces3:
    for i in range(x,x+w):
        for j in range(y,y+h):
            pointsf3.append([i,j])
    cv2.rectangle(image3, (x, y), (x+w, y+h), (255, 0, 0), 2)



print(image[0][0])
print(image2[0][0])
print(np.add(image[0][0],image2[0][0]))
print(np.divide(np.add(image[0][0],image2[0][0]),2).astype(int))

for i in range(0,300):
    for j in range(0,300):
        a=math.pow(gray[i][j],2)
        b=math.pow(gray2[i][j],2)

        y=np.add(a,b)

        z=math.sqrt(y)
        gray2[i][j] = np.divide(z,2).astype(int)

cv2.imshow("image",gray);
cv2.waitKey(0);


cv2.imshow("image3",gray3);
cv2.waitKey(0);
cv2.imshow("image2",gray2);
cv2.waitKey(0);
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)

    output = visualize_facial_landmarks(image, shape)
    #cv2.imshow("Image", output)
    cv2.waitKey(0)
