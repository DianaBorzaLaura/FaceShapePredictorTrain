from xml.dom import minidom

import ET as ET
import cv2
import dlib
import imutils
from imutils import face_utils
import xml.etree.ElementTree as ET


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 2)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class Facial_landmarks:

    def __init__(self, imgs, img_paths, shape_predictor, file):
        self.img_paths = img_paths
        self.imgs = imgs
        self.shape_predictor = shape_predictor
        self.face_lands = {}
        self.face_rects = {}
        self.file = file



    def detect_facial_landmarks(self, attr):
        if (attr == 'image'):
            for img in self.imgs:
                image = imutils.resize(img, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.shape_predictor(gray, 1)

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
        else:
            for img_path in self.img_paths:
                image = cv2.imread(img_path)
                image = imutils.resize(image, width=500)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = self.shape_predictor(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = self.shape_predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    self.face_rects[image] = [x, y, w, h]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    lands = []
                    for (j, (x, y)) in enumerate(shape):
                        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                        index = "{0:0=2d}".format(j)
                        lands.append([j, (x, y)])

                    self.face_lands[image] = lands



    def get_face_lands(self):
        return self.face_lands

    def get_photo_landmarks(self, photo):
        return self.face_lands[photo]

    def get_face_rects(self, img):
        return self.face_rects[img]

    def load_face_predictor(self, predictor):
        self.shape_predictor = predictor
        predictor = dlib.shape_predictor(predictor)

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
        (x, y, w, h) = self.get_face_rects(img)
        tree = ET.parse('trainings.xml')
        root = tree.getroot()

        start = ET.SubElement(root, 'image')
        box = ET.SubElement(start, 'box', top=str(x), left=str(y), width=str(w), height=str(h))

        res = [[1, (2, 3)], [2, (3, 4)]]
        for (j, (x, y)) in res:
            part = ET.SubElement(box, 'part', name=str(j), x=str(x), y=str(y))

        # print(ET.tostring(root).decode())
        indent(root)
        with open('trainings.xml', 'w') as f:
            f.write(ET.tostring(root, encoding='utf8').decode('utf8'))
