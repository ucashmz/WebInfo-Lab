import numpy as np
import math
import sys
import os
import csv
from numpy.core.fromnumeric import size

# https://github.com/csaluja/JupyterNotebooks-Medium

def rounding(rate, zero_one, one_two, two_three, three_four, four_five):
    if rate > two_three:
        if rate > three_four:
            if rate > four_five:
                return 5
            else:
                return 4
        else:
            return 3
    else:
        if rate > zero_one:
            if rate > one_two:
                return 2
            else:
                return 1
        else:
            return 0

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
        zero = np.zeros(maxuser + 1, dtype=int)
        for record in cs:
            origin[int(record[1])][int(record[0])] = int(record[2])
            if int(record[2]) != 0:
                nonzero[int(record[0])] += 1
            else:
                zero[int(record[0])] += 1
        # print(nonzero)
        # print(origin)
        # print(size(nonzero))
        zero = zero / (zero + nonzero + 0.00001) # zero-rate percentage
        print(zero.sum()/2185)
        return origin, nonzero, zero

    def getAverV(self, origin, nonzero):
        print("getAverV")
        sumV = origin.sum(axis=0)
        users = size(sumV)
        averV = np.zeros(users, dtype=float)
        for i in range(users):
            averV[i] = float(sumV[i] / nonzero[i])
            if math.isnan(averV[i]):
                averV[i] = float(2)
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
            sqrtV[y] = np.linalg.norm(newM[:,y])
        return sqrtV

    def getSim(self, newM, sqrtV, userid):
        print("getSim")
        simV = np.zeros(newM.shape[1], dtype=float)
        for y in range(newM.shape[1]):
            sumup = np.dot(newM[:,y], newM[:,userid])
            norm = sqrtV[y] *sqrtV[userid]
            if norm:
                simV[y] = sumup/norm
            else:
                simV[y] = 0
        return simV
    
    def predict(self, simV, newM, averV, userId, movieId, nonzero):
        kNear = []
        index = 0
        tmpV = np.copy(simV)
        while index < 5:
            maxSim = np.argmax(tmpV)
            if maxSim == userId:
                tmpV[maxSim] = -1
                continue
            if tmpV[maxSim] <= 0:
                break
            if newM[movieId][maxSim] != 0:
                kNear.append(maxSim)
                index += 1
            tmpV[maxSim] = -1
        # print(kNear)
        out = 0.0
        sumSim = 0.0
        if len(kNear) == 5:
            for item in kNear:
                # print(item, simV[item], newM[movieId][item])
                out += simV[item] * newM[movieId][item]
                sumSim += simV[item]
            out = out/sumSim + averV[userId]
        else:
            out = averV[userId]

        out = out * nonzero[userId]

        #if out > 5:
        #    out = 5
        #if out < 0:
        #    out = 0
        #out = rounding(out, roundingEdge[0], roundingEdge[1], roundingEdge[2], roundingEdge[3], roundingEdge[4])

        # print(userId, movieId, out, nonzero[userId])

        return out

    def recommend(self, newM, sqrtV, averV, zero):
        print("recommend")
        count = 0
        testDat = os.path.join(self.dataset_path, self.testData)
        if not os.path.exists(testDat):
            print("not found dataset, it should be WORKPATH/dataset/testing.dat")
            exit()
        with open(testDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))

        with open(os.path.join(self.out_path, self.testing_out), 'w', encoding='utf-8') as file:
            simV = self.getSim(newM, sqrtV, int(cs[0][0]))
            userId = int(cs[0][0])
            nonzero = np.power(1 - zero, 1.5) # 1.5 is a parameter
            print("Non zero = ", nonzero[userId])
            
            for record in cs:
                if int(record[0]) != userId:
                    userId = int(record[0])
                    simV = self.getSim(newM, sqrtV, int(record[0]))
                    print("Sim got.")
                    print("Non zero = ", nonzero[userId])

                out = self.predict(simV, newM, averV, userId, int(record[1]), nonzero)
                file.write(str(out)+"\n")

                count += 1
                if count % 1000 == 0:
                    print(count/1000)
        
        print(count)

    def validation(self, newM, sqrtV, averV, zero):
        np.set_printoptions(threshold=np.inf)
        print("recommend")
        result = []
        count = 0
        error = 0.0

        testDat = os.path.join(self.dataset_path, self.trainData)
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
            
            nonzero = np.power(1 - zero, 1.5) # 1.5 is a parameter
            
            out = self.predict(simV, newM, averV, userId, int(record[1]), nonzero)
            error += math.pow(float(record[2]) - out, 2)
            
            count += 1
            if count % 1000 == 0:
                print(count/1000)
            if count == 10000: 
                break

        print("RMSE: ", math.sqrt(error/count))


        

class Knn_2:
    def __init__(self, config):
        self.dir = config['WORKPATH']
        self.dataset_path = config['dataset_path']
        self.testData = config['testData']
        self.trainData = config['trainData']
        self.userAve = config['userAve']
        self.testing_out = config['testing_out']
        self.out_path = config["out_path"]

    def init(self):
        '''
        output: data: dictionary of dictionary, data[USER][MOVIE] = RATE
        '''
        print("genMTrain")
        trainDat = os.path.join(self.dataset_path, self.trainData)
        data = dict()
        if not os.path.exists(trainDat):
            print("not found dataset, it should be WORKPATH/dataset/training.dat")
            exit()
        # file_path = os.path.join(self.dataset_path, filename)
        with open(trainDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        for record in cs:
            if int(record[2]) != 0:
                if record[0] not in data:
                    data[record[0]] = dict()
                data[record[0]][record[1]] = int(record[2])
        return data

    def getAverV(self, data):
        '''
        output: averV: np array
        '''
        averV = np.zeros(2185)
        for user in data.keys():
            averV[int(user)] = np.mean(list(data[user].values()))
        print("averV:", averV)
        return averV

    def getNormalized(self, data, averV, sqrtV):
        '''
        normalize data with averV and sqrtV
        '''
        print("getNormalized")
        for user in data.keys():
            for movie in data[user].keys():
                data[user][movie] = data[user][movie] - averV[int(user)]
        return data
        # print(data)

    def getSqrtV(self, data, averV):
        '''
        output: sqrtV: np array
        '''
        sqrtV = np.zeros(2185)
        for user in data.keys():
            movies = list(data[user].values())
            aver = averV[int(user)]
            movies = [movie - aver for movie in movies]
            # print(movies)
            sqrtV[int(user)] = np.sqrt(np.mean(np.square(movies)))
        print("sqrtV:", sqrtV)
        return sqrtV

    def getSim(self, data, userid):
        '''
        Sparse Matrix?
        Only if a movie is rated by both users, calculate the square sum
        '''
        print("getSim, userId =", userid)
        simV = np.zeros(2185)
        userid = str(userid)
        if userid not in data:
            return simV
        for user in data.keys():
            if len(data[userid]) > len(data[user]):
                a = data[userid]
                b = data[user]
            else:
                a = data[user]
                b = data[userid]
            sumAA = 0
            sumBB = 0
            sumAB = 0
            for movie in b.keys():
                if movie in a:
                    sumAA += a[movie]*a[movie]
                    sumAB += a[movie]*b[movie]
                    sumBB += b[movie]*b[movie]
            if sumAA > 0 and sumBB > 0:
                simV[int(user)] = sumAB / np.sqrt(sumAA) / np.sqrt(sumBB)
        return simV

    def dealMovieV(self, movieV):
        v1 = 0
        v2 = 0
        v3 = 0
        v4 = 0
        v5 = 0
        for movie in movieV:
            v1 += (movie - 1)*(movie - 1)
            v2 += (movie - 2)*(movie - 2)
            v3 += (movie - 3)*(movie - 3)
            v4 += (movie - 4)*(movie - 4)
            v5 += (movie - 5)*(movie - 5)
        v = [v1, v2, v3, v4, v5]
        ret = np.argmin(v) + 1
        return ret
        # return np.mean(movieV)

    def getAverMovie(self):
        print('getAverMovie')
        trainDat = os.path.join(self.dataset_path, self.trainData)
        if not os.path.exists(trainDat):
            print("not found dataset, it should be WORKPATH/dataset/training.dat")
            exit()
        with open(trainDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        maxmovie = int(cs[0][0])
        minmovie = int(cs[0][1])
        for record in cs:
            if int(record[1]) > maxmovie:
                maxmovie = int(record[1])
            if int(record[1]) < minmovie:
                minmovie = int(record[1])
        origin = dict()
        for record in cs:
            if int(record[2]) != 0:
                if record[1] not in origin:
                    origin[record[1]] = dict()
                origin[record[1]][record[0]] = int(record[2])
        averMovie = np.zeros(maxmovie + 1)
        for movie in origin.keys():
            averMovie[int(movie)] = self.dealMovieV(list(origin[movie].values()))
        return averMovie

    def recommend(self, data, sqrtV, averV, averMovie):
        print("recommend")
        testDat = os.path.join(self.dataset_path, self.testData)
        if not os.path.exists(testDat):
            print("not found dataset, it should be WORKPATH/dataset/testing.dat")
            exit()
        with open(testDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        simV = self.getSim(data, int(cs[0][0]))
        userId = int(cs[0][0])
        for record in cs:
            if int(record[0]) != userId:
                userId = int(record[0])
                simV = self.getSim(data, int(record[0]))
            kNear = []
            tmpV = simV.tolist()
            index = 0
            while index < 15:
                maxSim = tmpV.index(max(tmpV))
                # bug fix here, allow using less than 5 neighbors if similarity already less than 0
                if max(tmpV) <= 0:
                    break
                # print(data[str(maxSim)])
                if record[1] in data[str(maxSim)]:
                    kNear.append(maxSim)
                    index += 1
                tmpV[maxSim] = -1
            # print(kNear)
            # print(averV[userId], sqrtV[userId])
            out = 0.0
            sumSim = 0.0
            for user in kNear:
                out += simV[user] * data[str(user)][record[1]]
                # print(data[str(user)][record[1]])
                sumSim += simV[user]
            if sumSim == 0:
                # out = averV[userId]
                out = averMovie[int(record[1])]
            else:
                out = out/sumSim + averV[userId]
                if out > 5:
                    out = 5
                if out < 0:
                    out = 0
                out = round(out)
                if (str(out)).find('.') != -1:
                    out = round(out)
            # print(userId, int(record[1]), out)

            with open(os.path.join(self.out_path, self.testing_out), 'a', encoding='utf-8') as file:
                file.write(str(out))
                file.write('\n')

    def validation(self, data, sqrtV, averV):
        print("validation")
        error = 0
        count = 0
        # run on training data, record with rating != 0
        testDat = os.path.join(self.dataset_path, self.trainData)
        if not os.path.exists(testDat):
            print("not found dataset, it should be WORKPATH/dataset/training.dat")
            exit()
        with open(testDat, 'r', encoding='utf-8') as csvfile:
            cs = list(csv.reader(csvfile))
        simV = self.getSim(data, int(cs[0][0]))
        userId = int(cs[0][0])
        for record in cs:
            if int(record[2]) != 0:
                if int(record[0]) != userId:
                    print("same")
                    userId = int(record[0])
                    simV = self.getSim(data, int(record[0]))
                kNear = []
                tmpV = simV.tolist()
                index = 0
                while index < 5:
                    maxSim = tmpV.index(max(tmpV))
                    if max(tmpV) <= 0:
                        break
                    # print(data[str(maxSim)])
                    if record[1] in data[str(maxSim)] and maxSim != userId:
                        kNear.append(maxSim)
                        index += 1
                    tmpV[maxSim] = -1
                # print(kNear)
                # print(averV[userId], sqrtV[userId])
                out = 0.0
                sumSim = 0.0
                for user in kNear:
                    out += simV[user] * data[str(user)][record[1]]
                    # print(data[str(user)][record[1]])
                    sumSim += simV[user]
                if sumSim == 0:
                    out = 3.8
                else:
                    out = out/sumSim * sqrtV[userId] + averV[userId]
                    if out > 5:
                        out = 5
                    if out < 1:
                        out = 1
                # print(userId, int(record[1]), out, record[2])
                error += math.pow(float(record[2]) - out, 2)
                count += 1
                if count % 1000 == 0:
                    print(count/1000)
                if count == 5000:  # just run 5000 ratings
                    break

        print("RMSE: ", math.sqrt(error/count))
