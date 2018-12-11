#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to use dlib's implementation of the paper:
#   One Millisecond Face Alignment with an Ensemble of Regression Trees by
#   Vahid Kazemi and Josephine Sullivan, CVPR 2014
#
#   In particular, we will train a face landmarking model based on a small
#   dataset and then evaluate it.  If you want to visualize the output of the
#   trained model on some images then you can run the
#   face_landmark_detection.py example program with predictor.dat as the input
#   model.
#
#   It should also be noted that this kind of model, while often used for face
#   landmarking, is quite general and can be used for a variety of shape
#   prediction tasks.  But here we demonstrate it only on a simple face
#   landmarking task.
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import os
import sys
import glob

import cv2
import dlib


# In this example we are going to train a face detector based on the small
# faces dataset in the examples/faces directory.  This means you need to supply
# the path to this faces folder as a command line argument so we will know
# where it is.
# if len(sys.argv) != 2:
#     print(
#         "Give the path to the examples/faces directory as the argument to this "
#         "program. For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./train_shape_predictor.py ../examples/faces")
#     exit()
# faces_folder = sys.argv[1]
from imutils import face_utils

options = dlib.shape_predictor_training_options()
# Now make the object responsible for training the model.
# This algorithm has a bunch of parameters you can mess with.  The
# documentation for the shape_predictor_trainer explains all of them.
# You should also read Kazemi's paper which explains all the parameters
# in great detail.  However, here I'm just setting three of them
# differently than their default values.  I'm doing this because we
# have a very small dataset.  In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
options.oversampling_amount = 300
# I'm also reducing the capacity of the model by explicitly increasing
# the regularization (making nu smaller) and by using trees with
# smaller depths.
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

# dlib.train_shape_predictor() does the actual training.  It will save the
# final predictor to predictor.dat.  The input is an XML file that lists the
# images in the training dataset and also contains the positions of the face
# parts.
training_xml_path = os.path.join("trainings.xml")
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

# Now that we have a model we can test it.  dlib.test_shape_predictor()
# measures the average distance between a face landmark output by the
# shape_predictor and where it should be according to the truth data.
print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, "predictor.dat")))
# The real test is to see how well it does on data it wasn't trained on.  We
# trained it on a very small dataset so the accuracy is not extremely high, but
# it's still doing quite good.  Moreover, if you train it on one of the large
# face landmarking datasets you will obtain state-of-the-art results, as shown
# in the Kazemi paper.
testing_xml_path = os.path.join("testing.xml")
print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))

# Now let's use it as you would in a normal application.  First we will load it
# from disk. We also need to load a face detector to provide the initial
# estimate of the facial location.
predictor = dlib.shape_predictor("predictor.dat")
detector = dlib.get_frontal_face_detector()

# Now let's run the detector and shape_predictor over the images in the faces
# folder and display the results.

pred = open("predicted.xml", "w")
pred.write("<dataset>\n")
pred.write("<images>\n")

print("Showing detections and predictions on the images in the faces folder...")
win = dlib.image_window()
for f in glob.glob("faces/7285955@N06/*.jpg"):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    pred.write(f" <image file='{f}'>\n")
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        shape = face_utils.shape_to_np(shape)
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                           shape.part(1)))
        (x, y, w, h) = face_utils.rect_to_bb(d)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face #{}".format(k + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        pred.write(f"    <box top='{y}' left='{x}' width='{w}' height='{h}'>\n")
        for (j, (x, y)) in enumerate(shape):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
            index = "{0:0=2d}".format(j)
            pred.write(f"    <part name='{index}' x='{x}' y='{y}'/>\n")
        # Draw the face landmarks on the screen.

        pred.write(f"</box>\n")

        pred.write(f"</image>\n")
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        #win.add_overlay(shape)



    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

pred.write(f"</images>\n")
pred.write(f"</dataset>\n")
pred.close()