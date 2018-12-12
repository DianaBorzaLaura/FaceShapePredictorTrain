from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import glob
import re
import random
from xml.dom import minidom


class Facial_landmarks:

    def __init__(self, imgs, img_paths, shape_predictor):
        self.img_paths = img_paths
        self.imgs = imgs
        self.shape_predictor = shape_predictor
        self.face_lands = {}
        self.face_rects = {}



    def detect_facial_landmarks(self):
        for img in self.imgs:
            image = imutils.resize(img, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.shape_predictor(gray, 1)

            face_rects = []
            for (i, rect) in enumerate(rects):
                shape = self.shape_predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                self.face_rects[img] = [x,y,w,h]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


                cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                lands = []
                for (j, (x,y)) in enumerate(shape):
                    cv2.circle(image, (x,y), 1, (0, 0, 255), -1)
                    index = "{0:0=2d}".format(j)
                    lands.append([x,y])

                self.face_lands[img] = lands


    def get_face_lands(self):
        return self.face_lands

    def get_photo_landmarks(self, photo):
        return self.face_lands[photo]

    def load_face_predictor(self, predictor):
        self.shape_predictor = predictor

    def read_xml(self, file):
        l = []
        i = 0
        imgs = []
        xmldoc = minidom.parse(file)
        img = xmldoc.getElementsByTagName('image')
        for im in img:
            imgs.append(im.attributes['file'].value)
        itemlist = xmldoc.getElementsByTagName('part')
        j = 0
        for s in itemlist:

            if (i < 68):
                l.append([s.attributes['x'].value, s.attributes['y'].value])
                i += 1

            else:
                self.face_lands[imgs[j]] = l
                j += 1
                i = 0
                l = []


    def export_to_xml(self, img):
        sir = "<image file='img'>\n"
        sir += "\t<box top='>"
