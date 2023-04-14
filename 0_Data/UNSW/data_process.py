import pandas as pd
import torch
import numpy as np
import pickle
import argparse
import math

def dataset_distribute(dataframes, n_c, total_num):
    datalists = []
    data_per_client = math.floor(total_num/n_c)
    save_df = pd.DataFrame()
    pos = 0

    for df in dataframes:
        ids=0
        datalist=[]
        length = len(df)

        #It will stays in the same dataframe
        while ids < length:
            #When the data_per_client too big
            if (length-ids) < data_per_client or length < data_per_client:

                if len(save_df)!=0:
                    gap = data_per_client-len(save_df)

                    if gap<length:
                        save_df = pd.concat([save_df, df.iloc[ids:ids+gap, :]])
                        pos = pos + gap
                        ids = ids+gap
                        datalists.append(save_df)
                        #Empty the dataframe
                        save_df = pd.DataFrame()
                        continue

                    # Here is the case that continue adding the whole dataset to save_df
                    save_df = pd.concat([save_df, df.iloc[:, :]])
                    pos = pos + length
                    # If it comes to the last dataframe, output the save_df
                    if pos == total_num:
                        datalists.append(save_df)
                    break

                else:
                    #There is no dataframes in the buffer, creates a new dataframe to the save the stage of data. After this, start from next dataframe
                    df_temp = df.iloc[ids:ids+(length-ids), :]
                    save_df = df_temp
                    pos = pos + (length-ids)
                    break

            else:
                if len(save_df)!=0:
                    gap = data_per_client-len(save_df)
                    save_df = pd.concat([save_df, df.iloc[ids:ids+gap, :]])
                    pos=pos+gap
                    ids = ids+gap
                    datalists.append(save_df)
                    #Empty the dataframe
                    save_df = pd.DataFrame()
                else:
                    df_temp = df.iloc[ids:ids+data_per_client, :]
                    ids = ids + data_per_client
                    pos = pos + data_per_client
                    datalists.append(df_temp)


    return datalists

def dataframe_to_tensor(datalists):
    tensor_list = []
    for df in datalists:
        tensor = torch.FloatTensor(df.values.astype(np.float32))
        tensor_list.append(tensor)

    return tensor_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fiveclient', action="store_true")
    parser.add_argument('--num_client', type=int, required=False)
    args = parser.parse_args()

    DATA_PATH = "./"
    df = pd.read_csv(DATA_PATH+"UNSW_2018_IoT_Botnet_Final_10_Best.csv",header=None, sep=';')
    df = df.dropna()


    A1 = df[df.iloc[:,18].str.match('Reconnaissance')]
    A2 = df[df.iloc[:,18].str.match('DoS')]
    A3 = df[df.iloc[:,18].str.match('DDoS')]
    B = df[df.iloc[:,18].str.match('Normal')]

    if (args.fiveclient):
        total = len(A1)
        iid = 0
        part_samp = int(0.5*total)
        df_A1 = A1.iloc[0:part_samp, 7:17]
        iid = iid + part_samp
        df_A4 = A1.iloc[iid:iid+part_samp, 8:17]

        total = len(A2)
        iid = 10000
        part_samp = int(0.01*total)
        df_A2 = A2.iloc[0:part_samp, 8:17]
        print(df_A2.shape)

        total = len(A3)
        iid = 0
        part_samp = int(0.05*total)
        df_A3 = A3.iloc[0:part_samp, 7:17]
        iid = iid + part_samp
        df_A5 = A3.iloc[iid:iid+part_samp, 8:17]

        total = len(B)
        part_samp = round(total/5)
        iid = 0
        df_B1 = B.iloc[0:part_samp, 8:17]
        iid = iid + part_samp
        df_B2 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B3 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B4 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B5 = B.iloc[iid:iid+part_samp, 8:17]


        tB1 = torch.FloatTensor(df_B1.values.astype(np.float32))
        tA1 = torch.FloatTensor(df_A1.values.astype(np.float32))
        tB2 = torch.FloatTensor(df_B2.values.astype(np.float32))
        tA2 = torch.FloatTensor(df_A2.values.astype(np.float32))
        tB3 = torch.FloatTensor(df_B3.values.astype(np.float32))
        tA3 = torch.FloatTensor(df_A3.values.astype(np.float32))
        tB4 = torch.FloatTensor(df_B4.values.astype(np.float32))
        tA4 = torch.FloatTensor(df_A4.values.astype(np.float32))
        tB5 = torch.FloatTensor(df_B5.values.astype(np.float32))
        tA5 = torch.FloatTensor(df_A5.values.astype(np.float32))
        print(tB1.shape)
        print(tA1.shape)
        print(tB2.shape)
        print(tA2.shape)
        print(tB3.shape)
        print(tA3.shape)
        print(tB4.shape)
        print(tA4.shape)
        print(tB5.shape)
        print(tA5.shape)
        d0 = [tB1, tB2, tB3, tB4, tB5]
        d1 = [tA1, tA2, tA3, tA4, tA5]

        #Creare Pickle files for the dataset being used
        with open("./d0.p", "wb") as handle:
            pickle.dump(d0, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open("./d1.p", "wb") as handle:
            pickle.dump(d1, handle, protocol = pickle.HIGHEST_PROTOCOL)

    else:
        total = len(A1)
        iid = 0
        part_samp = int(0.4*total)
        df_A1 = A1.iloc[0:part_samp, 7:17]
        iid = iid + part_samp
        df_A4 = A1.iloc[iid:iid+part_samp, 8:17]

        total = len(A2)
        iid = 10000
        part_samp = int(0.5*total)
        df_A2 = A2.iloc[0:part_samp, 8:17]
        print(df_A2.shape)

        total = len(A3)
        iid = 0
        part_samp = int(0.4*total)
        df_A3 = A3.iloc[0:part_samp, 7:17]
        iid = iid + part_samp
        df_A5 = A3.iloc[iid:iid+part_samp, 8:17]

        total = len(B)
        part_samp = round(total/5)
        iid = 0
        df_B1 = B.iloc[0:part_samp, 8:17]
        iid = iid + part_samp
        df_B2 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B3 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B4 = B.iloc[iid:iid+part_samp, 8:17]
        iid = iid + part_samp
        df_B5 = B.iloc[iid:iid+part_samp, 8:17]

        number_b_data = len(df_B1)+len(df_B2)+len(df_B3)+len(df_B4)+len(df_B5)
        number_a_data = len(df_A1)+len(df_A2)+len(df_A3)+len(df_A4)+len(df_A5)

        datalists_benign = dataset_distribute([df_B1,df_B2,df_B3,df_B4,df_B5], args.num_client, number_b_data)
        datalists_attack = dataset_distribute([df_A1,df_A2,df_A3,df_A4,df_A5], args.num_client, number_a_data)
        d0 = dataframe_to_tensor(datalists_benign)
        d1 = dataframe_to_tensor(datalists_attack)

        #Creare Pickle files for the dataset being used
        with open("./d0_clients.p", "wb") as handle:
            pickle.dump(d0, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open("./d1_clients.p", "wb") as handle:
            pickle.dump(d1, handle, protocol = pickle.HIGHEST_PROTOCOL)
