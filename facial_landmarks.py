from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import re
import random



predictor_path = "shape_predictor_68_face_landmarks.dat"
#predictor_path = "detector.svm"
image_path = "images/imagine.jpg"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#predictor = dlib.simple_object_detector(predictor_path)

out = open("trainings.xml", "w")
test = open("testing.xml", "w")
out.write("<dataset>\n")
test.write("<dataset>\n")
out.write("<images>\n")
test.write("<images>\n")

for image_path in glob.glob("faces/7285955@N06/*.jpg"):
    r = random.randint(0,100)
    print(r)
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    rects = detector(gray, 1)
    #rects = detector(image)

    if (r > 10):
        out.write(f" <image file='{image_path}'>\n")

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            out.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (j, (x, y)) in enumerate(shape):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                index = "{0:0=2d}".format(j)
                out.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")


            out.write(f"</box>\n")

        out.write(f"</image>\n")
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)



    else:
        test.write(f" <image file='{image_path}'>\n")

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            test.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")

            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for (j, (x, y)) in enumerate(shape):
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                index = "{0:0=2d}".format(j)
                test.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")
            test.write(f"</box>\n")

        test.write(f"</image>\n")
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)


out.write(f"</images>\n")
out.write(f"</dataset>\n")
out.close()

test.write(f"</images>\n")
test.write(f"</dataset>\n")
test.close()