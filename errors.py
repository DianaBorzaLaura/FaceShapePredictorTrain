from math import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import xml.etree.ElementTree as ET
import xml.etree.ElementTree
from xml.dom import minidom


def read_xml(file):

    l = []
    grtr = {}
    i = 0
    imgs = []
    xmldoc = minidom.parse(file)
    img = xmldoc.getElementsByTagName('image')
    for im in img:
        #print(im.attributes['file'].value)
        imgs.append(im.attributes['file'].value)
    itemlist = xmldoc.getElementsByTagName('part')
    j = 0
    for s in itemlist:
        # print(s.attributes['x'].value)
        # print(s.attributes['y'].value)
        # print(imgs[i])
        # print(i)

        if (i < 68):
            l.append([s.attributes['x'].value, s.attributes['y'].value])
            i += 1

        else:
            #print(imgs[j])
            grtr[imgs[j]] = l
            j+=1
            i = 0
            l = []

    #print(grtr)
    return grtr

#
# read_xml('trainings.xml')
# read_xml('predicted.xml')

def mean_error(no_pt, pred, grtr):
    #no_pt: list of the points
    #pred: list of lists of tuples: [[(x1,y1), ..., (xn, yn)], [],...[]] for every photo
    #grtr: list of lists of tuples

    #print(pred['faces/7285955@N06/coarse_tilt_aligned_face.2050.9486768763_e52727c632_o.jpg'])
    err = 0
    means = []
    variances = []
    N = len(pred)
    sum_err = 0
    sum_sqr = 0
    for i in no_pt:
        for photo in grtr:
            err = sqrt((int(pred[photo][i][0])-int(grtr[photo][i][0]))**2 + (int(pred[photo][i][1])-int(grtr[photo][i][1]))**2)
            sum_err += err
            sum_sqr += err**2


    mean = err/N
    means.append(mean)
    variances.append((sum_sqr - N*mean**2)/(N-1))

    return (means, variances)

# no_pt = [2,3]
# pred = read_xml('predicted.xml')
# grtr = read_xml('testing.xml')
# print(mean_error(no_pt, pred, grtr))
# def plots(res):
#     data = []
#     for i in res[1].keys():
#         data.append(res[1][i])
#
#     data_sorted = np.sort(data)
#     p = 1. * arange(len(data)) / (len(data) - 1)
#
#     fig = figure()
#     ax1 = fig.add_subplot(121)
#     ax1.plot(p, data_sorted)
#     ax1.set_xlabel('$p$')
#     ax1.set_ylabel('$x$')
#
#     ax2 = fig.add_subplot(122)
#     ax2.plot(data_sorted, p)
#     ax2.set_xlabel('$p$')
#     ax2.set_ylabel('$x$')
#     plt.show()


#res = mean_error(no_pt, pred, grtr)
#print(res)
# plots(res)

sir = ""
sir += "asbsdh"
print(sir)







