import numpy as np
# from numpy.core.arrayprint import dtype_is_implied
# import pandas as pd
# import sklearn.metrics as metrics
# from sklearn.neighbors import NearestNeighbors
# from scipy.spatial.distance import correlation, cosine
# from sklearn.metrics import pairwise_distances
# from sklearn.metrics import mean_squared_error
import math
# import sys
import os
# from contextlib import contextmanager
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

    def genFromTrain(self):
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
        csvfile.close()
        return origin, nonzero
    
    def getAver(self, origin, nonzero):
        sumV = origin.sum(axis=0)
        users = size(sumV)
        averV = np.zeros(users, dtype=float)
        for i in range(users):
            averV[i] = float(sumV[i] / nonzero[i])
            if math.isnan(averV[i]):
                averV[i] = float(0)
        return averV

'''
M = np.asarray([[3, 7, 4, 9, 9, 7],
                [7, 0, 5, 3, 8, 8],
                [7, 5, 5, 0, 8, 4],
                [5, 6, 8, 5, 9, 8],
                [5, 8, 8, 8, 10, 9],
                [7, 7, 0, 4, 7, 8]])
M = pd.DataFrame(M)

# declaring k,metric as global which can be changed by the user later
global k, metric
k = 4
metric = 'cosine'  # can be changed to 'correlation' for Pearson correlation similaries

# get cosine similarities for ratings matrix M; pairwise_distances returns the distances between ratings and hence
# similarities are obtained by subtracting distances from 1
cosine_sim = 1-pairwise_distances(M, metric="cosine")


# Cosine similarity matrix
pd.DataFrame(cosine_sim)


# get pearson similarities for ratings matrix M
pearson_sim = 1-pairwise_distances(M, metric="correlation")

# Pearson correlation similarity matrix
pd.DataFrame(pearson_sim)

# This function finds k similar users given the user_id and ratings matrix M
# Note that the similarities are same as obtained via using pairwise_distances


def findksimilarusers(user_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(
        ratings.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()
    # print('{0} most similar users for User {1}:\n',k,user_id)
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue
        else:
            pass
            # print('{0}: User {1}, with similarity of {2}', i, indices.flatten()[i]+1, similarities.flatten()[i])
    return similarities, indices


similarities, indices = findksimilarusers(1, M, metric='cosine')
similarities, indices = findksimilarusers(1, M, metric='correlation')


def predict_userbased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = 0
    # similar users based on cosine similarity
    similarities, indices = findksimilarusers(user_id, ratings, metric, k)
    # to adjust for zero based indexing
    mean_rating = ratings.loc[user_id-1, :].mean()
    sum_wt = np.sum(similarities)-1
    product = 1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue
        else:
            ratings_diff = ratings.iloc[indices.flatten(
            )[i], item_id-1]-np.mean(ratings.iloc[indices.flatten()[i], :])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product

    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    # print('\nPredicted rating for user', user_id, '-> item', item_id, ':', prediction)
    return prediction


predict_userbased(3, 4, M)


def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities = []
    indices = []
    ratings = ratings.T
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(
        ratings.iloc[item_id-1, :].values.reshape(1, -1), n_neighbors=k+1)
    similarities = 1-distances.flatten()
    # print '{0} most similar items for item {1}:\n'.format(k,item_id)
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue
        else:
            pass
            # print '{0}: Item {1} :, with similarity of {2}'.format(i,indices.flatten()[i]+1, similarities.flatten()[i])

    return similarities, indices


similarities, indices = findksimilaritems(3, M)
# This function predicts the rating for specified user-item combination based on item-based approach


def predict_itembased(user_id, item_id, ratings, metric=metric, k=k):
    prediction = wtd_sum = 0
    # similar users based on correlation coefficients
    similarities, indices = findksimilaritems(item_id, ratings)
    sum_wt = np.sum(similarities)-1
    product = 1

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == item_id:
            continue
        else:
            product = ratings.iloc[user_id-1,
                                   indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))
    # print '\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction)

    return prediction


prediction = predict_itembased(1, 3, M)

# This function is used to compute adjusted cosine similarity matrix for items


def computeAdjCosSim(M):
    sim_matrix = np.zeros((M.shape[1], M.shape[1]))
    M_u = M.mean(axis=1)  # means

    for i in range(M.shape[1]):
        for j in range(M.shape[1]):
            if i == j:

                sim_matrix[i][j] = 1
            else:
                if i < j:

                    sum_num = sum_den1 = sum_den2 = 0
                    for k, row in M.loc[:, [i, j]].iterrows():

                        if ((M.loc[k, i] != 0) & (M.loc[k, j] != 0)):
                            num = (M[i][k]-M_u[k])*(M[j][k]-M_u[k])
                            den1 = (M[i][k]-M_u[k])**2
                            den2 = (M[j][k]-M_u[k])**2

                            sum_num = sum_num + num
                            sum_den1 = sum_den1 + den1
                            sum_den2 = sum_den2 + den2

                        else:
                            continue

                    den = (sum_den1**0.5)*(sum_den2**0.5)
                    if den != 0:
                        sim_matrix[i][j] = sum_num/den
                    else:
                        sim_matrix[i][j] = 0

                else:
                    sim_matrix[i][j] = sim_matrix[j][i]

    return pd.DataFrame(sim_matrix)


adjcos_sim = computeAdjCosSim(M)
adjcos_sim

# This function finds k similar items given the item_id and ratings matrix M


def findksimilaritems_adjcos(item_id, ratings, k=k):

    sim_matrix = computeAdjCosSim(ratings)
    similarities = sim_matrix[item_id -
                              1].sort_values(ascending=False)[:k+1].values
    indices = sim_matrix[item_id-1].sort_values(ascending=False)[:k+1].index

    # print '{0} most similar items for item {1}:\n'.format(k,item_id)
    for i in range(0, len(indices)):
        if indices[i]+1 == item_id:
            continue

        else:
            pass
            # print '{0}: Item {1} :, with similarity of {2}'.format(i,indices[i]+1, similarities[i])

    return similarities, indices


similarities, indices = findksimilaritems_adjcos(3, M)


# This function predicts the rating for specified user-item combination for adjusted cosine item-based approach
# As the adjusted cosine similarities range from -1,+1, sometimes the predicted rating can be negative or greater than max value
# Hack to deal with this: Rating is set to min if prediction is negative, Rating is set to max if prediction is above max
def predict_itembased_adjcos(user_id, item_id, ratings):
    prediction = 0

    # similar users based on correlation coefficients
    similarities, indices = findksimilaritems_adjcos(item_id, ratings)
    sum_wt = np.sum(similarities)-1

    product = 1
    wtd_sum = 0
    for i in range(0, len(indices)):
        if indices[i]+1 == item_id:
            continue
        else:
            product = ratings.iloc[user_id-1, indices[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))
    if prediction < 0:
        prediction = 1
    elif prediction > 10:
        prediction = 10
    print('\nPredicted rating for user', user_id,
          '-> item', item_id, ':', prediction)

    return prediction
# prediction=predict_itembased_adjcos(3,4,M)


adjcos_sim

# This function utilizes above function to recommend items for selected approach. Recommendations are made if the predicted
# rating for an item is greater than or equal to 6, and the items has not been rated already


def recommendItem(user_id, item_id, ratings):
    if user_id < 1 or user_id > 6 or type(user_id) is not int:
        print('Userid does not exist. Enter numbers from 1-6')
    else:
        ids = ['User-based CF (cosine)', 'User-based CF (correlation)', 'Item-based CF (cosine)',
               'Item-based CF (adjusted cosine)']

        
        # approach = widgets.Dropdown(options=ids, value=ids[0],
        #                        description='Select Approach', width='500px')
        
        # def on_change(change):
        #     prediction = 0
        #     clear_output(wait=True)
        #     if change['type'] == 'change' and change['name'] == 'value':            
        #         if (approach.value == 'User-based CF (cosine)'):
        #             metric = 'cosine'
        #             prediction = predict_userbased(user_id, item_id, ratings, metric)
        #         elif (approach.value == 'User-based CF (correlation)')  :                       
        #             metric = 'correlation'               
        #             prediction = predict_userbased(user_id, item_id, ratings, metric)
        #         elif (approach.value == 'Item-based CF (cosine)'):
        #             prediction = predict_itembased(user_id, item_id, ratings)
        #         else:
        #             prediction = predict_itembased_adjcos(user_id,item_id,ratings)

        #         if ratings[item_id-1][user_id-1] != 0: 
        #             print('Item already rated')
        #         else:
        #             if prediction>=6:
        #                 print('\nItem recommended')
        #             else:
        #                 print('Item not recommended')

        # approach.observe(on_change)
        # display(approach)
        

# check for incorrect entries
# recommendItem(-1,3,M)
# recommendItem(3,4,M)


# This is a quick way to temporarily suppress stdout in particular code section
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# This is final function to evaluate the performance of selected recommendation approach and the metric used here is RMSE
# suppress_stdout function is used to suppress the print outputs of all the functions inside this function. It will only print
# RMSE values



def evaluateRS(ratings):
    ids = ['User-based CF (cosine)','User-based CF (correlation)','Item-based CF (cosine)','Item-based CF (adjusted cosine)']
    approach = widgets.Dropdown(options=ids, value=ids[0],description='Select Approach', width='500px')
    n_users = ratings.shape[0]
    n_items = ratings.shape[1]
    prediction = np.zeros((n_users, n_items))
    prediction= pd.DataFrame(prediction)
    def on_change(change):
        # clear_output(wait=True)
        with suppress_stdout():
            if change['type'] == 'change' and change['name'] == 'value':            
                if (approach.value == 'User-based CF (cosine)'):
                    metric = 'cosine'
                    for i in range(n_users):
                        for j in range(n_items):
                            prediction[i][j] = predict_userbased(i+1, j+1, ratings, metric)
                elif (approach.value == 'User-based CF (correlation)')  :                       
                    metric = 'correlation'               
                    for i in range(n_users):
                        for j in range(n_items):
                            prediction[i][j] = predict_userbased(i+1, j+1, ratings, metric)
                elif (approach.value == 'Item-based CF (cosine)'):
                    for i in range(n_users):
                        for j in range(n_items):
                            prediction[i][j] = predict_userbased(i+1, j+1, ratings)
                else:
                    for i in range(n_users):
                        for j in range(n_items):
                            prediction[i][j] = predict_userbased(i+1, j+1, ratings)
              
        MSE = mean_squared_error(prediction, ratings)
        RMSE = round(sqrt(MSE),3)
        # print "RMSE using {0} approach is: {1}".format(approach.value,RMSE)
              
    approach.observe(on_change)
    # display(approach)
evaluateRS(M)
'''
