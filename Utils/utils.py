
import numpy as np
import math
import torch
from sklearn import preprocessing
import torch.utils.data as Data
from sklearn.preprocessing import OneHotEncoder


def normalize(df):
    x = df.numpy() #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = torch.from_numpy(x_scaled).float()
    return x_scaled



def normalize_UNSW(training_data, xtst):
    trainX = []
    testX = []
    for x in training_data:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        x_scaled = torch.from_numpy(x_scaled).float()
        trainX.append(x_scaled)

    for x in xtst:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        x_scaled = torch.from_numpy(x_scaled).float()
        testX.append(x_scaled)

    return trainX, testX


def label_encoder(df):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df)
    print(enc.categories_)
    df_array = enc.transform(df).toarray() #Encode the classes to a binary array
    return df_array


# Wrape the dataset
def data_wrapper(x, y):
    #train_X = torch.from_numpy(x.float32)
    #train_Y = torch.from_numpy(y.float32)
    train_X = x
    train_Y = y

    #Wrap the dataset into the tensor dataset
    dataset = Data.TensorDataset(train_X, train_Y)


    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=False
    )
    batch, labels = next(iter(loader))
    print(batch.shape)
    return loader

def getvariance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)


# Generate a random network graph for k nodes
def create_network(num_client):
    adj_list = []
    for i in range(num_client):
        ad = [random.randint(0,1) for _ in range(num_client)]
        ad[i] = 0
        for j in range(len(adj_list)):
            ad[j] = adj_list[j][i]

        print(sum(ad))
        adj_list.append(ad)

    return adj_list



def create_list(num_list):
    name_list = []
    for i in range(num_list):
        li = []
        name_list.append(li)

    return name_list

def masks(size, c):
    a = torch.ones(size)
    a = (1/c) * a
    a = torch.bernoulli(a)

    return a
