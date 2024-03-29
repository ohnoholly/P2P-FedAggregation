from Utils import data_distribute
from Utils import utils
from Utils import models
from Utils import train
from Core import *
import pandas as pd
import torch
import pickle
import numpy as np
import random
import argparse

if __name__ == "__main__":

    # Seed
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--balanced', action="store_true")
    parser.add_argument('--fiveclient', action="store_true")
    parser.add_argument('--rebalancer', type=int)
    parser.add_argument('--attack_mode', type=int, required=False)
    parser.add_argument('--num_ads', type=int, required=False)
    parser.add_argument('--random_network', action="store_true")
    parser.add_argument('--num_client',  type=int, required=False)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    PATH = "0_Data/"+args.train_dataset+"/"
    if args.fiveclient:
        d0 = pickle.load(open(PATH+'d0.p', 'rb'))
        d1 = pickle.load(open(PATH+'d1.p', 'rb'))
    else:
        d0 = pickle.load(open(PATH+'d0_clients.p', 'rb'))
        d1 = pickle.load(open(PATH+'d1_clients.p', 'rb'))


    if (args.train_dataset == "IoT"):
        model =models.Classifier_nonIID()
        if args.balanced:
            frac0 = torch.tensor([0.4, 0.5, 0.85, 0.3, 0.4])
            frac1 = torch.tensor([0.4, 0.5, 0.95, 0.3, 0.45])
            #Test dataset fraction for each user
            frtst0 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
            frtst1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

            frac_tr0 = 0.7 #fraction of training data
            frac_tr1 = 0.7

            frac_tst0 = 0.1 #fraction of testing data
            frac_tst1 = 0.1

        else:
            #Create non_IID data split and normalize data for each client
            frac0 = torch.tensor([0.3, 0.5, 0.4, 0.2, 0.3])
            frac1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
            #Test dataset fraction for each user
            frtst0 = torch.tensor([0.4, 0.4, 0.3, 0.4, 0.3])
            frtst1 = torch.tensor([0.25, 0.2, 0.25, 0.2, 0.25])

            frac_tr0 = 0.4 #fraction of training data
            frac_tr1 = 0.02

            frac_tst0 = 0.3 #fraction of testing data
            frac_tst1 = 0.05



    if (args.train_dataset == "UNSW"):
        model =models.Classifier_nonIIDUNSW()
        if args.balanced:
            frac0 = torch.tensor([0.4, 0.5, 0.85, 0.3, 0.4])
            frac1 = torch.tensor([0.4, 0.5, 0.95, 0.3, 0.45])
            #Test dataset fraction for each user
            frtst0 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
            frtst1 = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

            frac_tr0 = 0.7 #fraction of training data
            frac_tr1 = 0.7

            frac_tst0 = 0.1 #fraction of testing data
            frac_tst1 = 0.1

        else:
            #Create non_IID data split and normalize data for each client
            frac0 = torch.tensor([0.3, 0.5, 0.4, 0.5, 0.4])
            frac1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1])
            #Test dataset fraction for each user
            frtst0 = torch.tensor([0.4, 0.4, 0.3, 0.4, 0.3])
            frtst1 = torch.tensor([0.26, 0.2, 0.23, 0.2, 0.23])

            frac_tr0 = 1 #fraction of training data
            frac_tr1 = 0.5

            frac_tst0 = 0.3 #fraction of testing data
            frac_tst1 = 0.5



    training_data, training_labels,global_testdata, g_testlabels = data_distribute.getData_nonIID(frac0,frac1,frtst0, frtst1, frac_tr0,frac_tr1, frac_tst0,frac_tst1, d0,d1)
    print("Training/test data splits finished")
    training_data = data_distribute.normalizeData_nonIID(training_data)
    global_testdata = utils.normalize(global_testdata)
    print("Data normalized")

    if args.num_ads is None:
        num_ads = ''
    else:
        num_ads = args.num_ads

    if args.attack_mode is None:
        attack_mode = 0
    else:
        attack_mode = args.attack_mode


    if args.random_network:
        # Create a random network
        adj_list = create_network(args.num_client)
    else:
        #Default incomplete graph
        adj_list=[[0,1,1,1,1], [1,0,1,1,0], [1,1,0,0,1], [1,1,0,0,0], [1,0,1,0,0]]


    if args.rebalancer ==1:
        # REBALANCER
        train.trainFA_imbalanced(model, training_data, training_labels, 0.8,
                                            100, 5, device, iid=False, gdata=global_testdata,
                                            glabel=g_testlabels)
    else:
        train.trainP2P(model, training_data, training_labels, 0.1, adj_list, 100,
                    5, 1, device, batch_size=128, iid=False,  gdata=global_testdata, glabel=g_testlabels,
                    balanced=True, attack_mode=attack_mode, num_ads=num_ads)
