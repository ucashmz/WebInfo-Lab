import numpy as np
# from numpy.core.arrayprint import dtype_is_implied
# import pandas as pd
import math
import sys
import os
import csv
from numpy.core.fromnumeric import size

# https://github.com/csaluja/JupyterNotebooks-Medium

class Knn:
    def __init__(self, config):
        self.dir = config['WORKPATH']
        self.dataset_path = config['dataset_path']
        self.testData = config['testData']
        self.trainData = config['trainData']
        self.userAve = config['userAve']
        self.testing_out = config['testing_out']
        self.out_path = config["out_path"]

    def genMTrain(self):
        print("genMTrain")
        trainDat = os.path.join(self.dataset_path, self.trainData)
        if not os.path.exists(trainDat):
            print("not found dataset, it should be WORKPATH/dataset/training.dat")
            exit()
        # file_path = os.path.join(self.dataset_path, filename)
        with open(trainDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        maxuser = int(cs[0][0])
        minuser = int(cs[0][1])
        maxmovie = int(cs[0][0])
        minmovie = int(cs[0][1])
        for record in cs:
            if int(record[0]) > maxuser:
                maxuser = int(record[0])
            if int(record[0]) < minuser:
                minuser = int(record[0])
            if int(record[1]) > maxmovie:
                maxmovie = int(record[1])
            if int(record[1]) < minmovie:
                minmovie = int(record[1])
        #         print(record)
        # print("user:", minuser, maxuser)
        # 0-2184
        # print("movie:", minmovie, maxmovie)
        # 0-74682

        origin = np.zeros((maxmovie + 1, maxuser + 1), dtype=int)
        nonzero = np.zeros(maxuser + 1, dtype=int)
        for record in cs:
            origin[int(record[1])][int(record[0])] = int(record[2])
            nonzero[int(record[0])] += 1
        # print(nonzero)
        # print(origin)
        # print(size(nonzero))
        return origin, nonzero

    def getAverV(self, origin, nonzero):
        print("getAverV")
        sumV = origin.sum(axis=0)
        users = size(sumV)
        averV = np.zeros(users, dtype=float)
        for i in range(users):
            averV[i] = float(sumV[i] / nonzero[i])
            if math.isnan(averV[i]):
                averV[i] = float(0)
        return averV

    def genNewM(self, origin, averV):
        print("genNewM")
        newM = np.zeros(origin.shape, dtype=float)
        for y in range(origin.shape[1]):
            for x in range(origin.shape[0]):
                if origin[x][y] != 0:
                    newM[x][y] = float(origin[x][y]) - averV[y]
        return newM

    def getSqrtV(self, newM):
        print("getSqrtV")
        sqrtV = np.zeros(newM.shape[1], dtype=float)
        for y in range(newM.shape[1]):
            sumup = float(0)
            for x in range(newM.shape[0]):
                sumup += np.square(newM[x][y])
            sqrtV[y] = np.sqrt(sumup)
        return sqrtV

    def getSim(self, newM, sqrtV, userid):
        print("getSim")
        simV = np.zeros(newM.shape[1], dtype=float)
        for y in range(newM.shape[1]):
            sumup = float(0)
            for x in range(newM.shape[0]):
                sumup += newM[x][y] * newM[x][userid]
            simV[y] = sumup/sqrtV[y]/sqrtV[userid]
        return simV

    def recommend(self, newM, sqrtV, averV):
        print("recommend")
        result = []
        testDat = os.path.join(self.dataset_path, self.testData)
        if not os.path.exists(testDat):
            print("not found dataset, it should be WORKPATH/dataset/testing.dat")
            exit()
        with open(testDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        simV = self.getSim(newM, sqrtV, int(cs[0][0]))
        userId = int(cs[0][0])
        for record in cs:
            if int(record[0]) != userId:
                print("same")
                userId = int(record[0])
                simV = self.getSim(newM, sqrtV, int(record[0]))
            kNear = []
            tmpV = simV.tolist()
            index = 0
            while index < 5:
                maxSim = tmpV.index(max(tmpV))
                if newM[int(record[1])][maxSim] != 0:
                    kNear.append(maxSim)
                    index += 1
                tmpV[maxSim] = -1
            print(kNear)
            out = 0.0
            sumSim = 0.0
            for item in kNear:
                out += simV[item] * newM[int(record[1])][item]
                sumSim += simV[item]
            out = out/sumSim + averV[userId]
            print(userId, int(record[1]), out)
            result.append(out)

        original_stdout = sys.stdout
        with open(os.path.join(self.out_path, self.testing_out), 'w', encoding='utf-8') as file:
            sys.stdout = file
            for item in result:
                print(item)
            sys.stdout = original_stdout