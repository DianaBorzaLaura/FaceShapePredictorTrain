import unittest
import xml

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+2)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def test():
    x = 1
    y = 2
    w = 3
    h = 4

    points = list(range(0,68))
    print(points)

    # tree = ET.parse('trainings.xml')
    # root = tree.getroot()
    # nodes = root.findall("images")
    # print(len(nodes))
    # root.remove(nodes[len(nodes) - 1])
    # print(ET.tostring(root).decode())

    # file = open('trainings.xml', 'w')
    # root = ET.Element("image")
    # img = 'adresa.imagine'
    # root.set('file', img)
    # box = ET.SubElement(root, 'box')
    # box.set('top', x)
    # box.set('left', y)
    # box.set('width', w)
    # box.set('height', h)
    #
    # res = [[1,(2,3)],[2,(3,4)]]
    # for (j, (x, y)) in res:
    #     part = ET.SubElement(box, 'part')
    #     part.set('name', j)
    #     part.set('x', x)
    #     part.set('y', y)
    #
    # tree = ET.ElementTree(root)
    # tree.write(file)
    # file.close()

    # tree = ET.parse('trainings.xml')
    # root = tree.getroot()
    #
    # start = ET.SubElement(root,'image')
    # box = ET.SubElement(start, 'box', top=str(x), left=str(y), width=str(w),height=str(h))
    #
    # res = [[1,(2,3)],[2,(3,4)]]
    # for (j, (x, y)) in res:
    #     part = ET.SubElement(box, 'part', name=str(j),x=str(x),y=str(y))
    #
    # #print(ET.tostring(root).decode())
    # indent(root)
    # with open('trainings.xml', 'w') as f:
    #     f.write(ET.tostring(root,encoding='utf8').decode('utf8'))

test()
