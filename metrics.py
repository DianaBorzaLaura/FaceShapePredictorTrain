from math import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import xml.etree.ElementTree as ET
import xml.etree.ElementTree
from xml.dom import minidom

class Metrics:

    def __init__(self, points, predicted, ground_truth, dimension):
        self.points = points
        self.predicted = predicted
        self.ground_truth = ground_truth
        self.dimension = dimension


    def mean_sqr_error(self):

        err = 0
        sum_err = 0
        sum_sqr = 0
        for i in self.points:
            for photo in self.ground_truth:
                pred = self.predicted[photo][i]
                grtr = self.ground_truth[photo][i]
                for dim in range(self.dimension):
                    err += int(pred[dim] - grtr[dim]) ** 2
                sum_err += sqrt(err)
                sum_sqr += abs(err)

        return (sum_err, sum_sqr)


    def mean_error(self):
        N = len(self.predicted)
        mean = self.mean_sqr_error()[0]/N
        return mean

    def variance(self):
        N = len(self.predicted)
        mean = self.mean_sqr_error()[0]/N
        variance = (self.mean_sqr_error()[1] - N*mean**2)/(N-1)

        return variance









