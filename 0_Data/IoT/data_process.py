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

    DATA_PATH = "./"

    df_B1 = pd.read_csv(DATA_PATH+"Provision_PT_838_Security_Camera/benign_traffic.csv")
    df_A1 = pd.read_csv(DATA_PATH+"Provision_PT_838_Security_Camera/ack.csv")
    df_B2 = pd.read_csv(DATA_PATH+"XCS7_1002_WHT_Security_Camera/benign_traffic.csv")
    df_A2 = pd.read_csv(DATA_PATH+"XCS7_1002_WHT_Security_Camera/ack.csv")
    df_B3 = pd.read_csv(DATA_PATH+"Danmini_Doorbell/benign_traffic.csv")
    df_A3 = pd.read_csv(DATA_PATH+"Danmini_Doorbell/ack.csv")
    df_B4 = pd.read_csv(DATA_PATH+"Ecobee_Thermostat/benign_traffic.csv")
    df_A4 = pd.read_csv(DATA_PATH+"Ecobee_Thermostat/udpplain.csv")
    df_B5 = pd.read_csv(DATA_PATH+"Philips_B120N10_Baby_Monitor/benign_traffic.csv")
    df_A5 = pd.read_csv(DATA_PATH+"Philips_B120N10_Baby_Monitor/ack.csv")





    parser = argparse.ArgumentParser()
    parser.add_argument('--fiveclient', action="store_true")
    parser.add_argument('--num_client', type=int)
    args = parser.parse_args()

    if (args.fiveclient):
        #Sample the data according to its portion (5 client)
        total = len(df_A1)
        print(total)
        part_samp = int(0.2 * total)
        print(part_samp)
        df_A1 = df_A1.iloc[0:part_samp, 0:115]
        total = len(df_B1)
        print(total)
        part_samp = int(0.12*total)
        print(part_samp)
        df_B1 = df_B1.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A2)
        print(total)
        part_samp = int(0.25*total)
        print(part_samp)
        df_A2 = df_A2.iloc[0:part_samp, 0:115]
        total = len(df_B2)
        print(total)
        part_samp = int(0.25*total)
        print(part_samp)
        df_B2 = df_B2.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A3)
        print(total)
        part_samp = int(0.1*total)
        print(part_samp)
        df_A3 = df_A3.iloc[0:part_samp, 0:115]
        total = len(df_B3)
        print(total)
        part_samp = int(0.25*total)
        print(part_samp)
        df_B3 = df_B3.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A4)
        print(total)
        part_samp = int(0.15*total)
        print(part_samp)
        df_A4 = df_A4.iloc[0:part_samp, 0:115]
        total = len(df_B4)
        print(total)
        part_samp = int(0.85*total)
        print(part_samp)
        df_B4 = df_B4.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A5)
        print(total)
        part_samp = int(0.1*total)
        print(part_samp)
        df_A5 = df_A5.iloc[0:part_samp, 0:115]
        total = len(df_B5)
        print(total)
        part_samp = int(0.05*total)
        print(part_samp)
        df_B5 = df_B5.iloc[0:part_samp, 0:115]

        #Feed to FL (5 clients)
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
        #Sample the data according to its portion (more clients)
        total = len(df_A1)
        print(total)
        part_samp = int(1 * total)
        print(part_samp)
        df_A1 = df_A1.iloc[0:part_samp, 0:115]
        total = len(df_B1)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_B1 = df_B1.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A2)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_A2 = df_A2.iloc[0:part_samp, 0:115]
        total = len(df_B2)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_B2 = df_B2.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A3)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_A3 = df_A3.iloc[0:part_samp, 0:115]
        total = len(df_B3)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_B3 = df_B3.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A4)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_A4 = df_A4.iloc[0:part_samp, 0:115]
        total = len(df_B4)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_B4 = df_B4.iloc[0:part_samp, 0:115]
        #Sample the data according to its portion
        total = len(df_A5)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_A5 = df_A5.iloc[0:part_samp, 0:115]
        total = len(df_B5)
        print(total)
        part_samp = int(1*total)
        print(part_samp)
        df_B5 = df_B5.iloc[0:part_samp, 0:115]

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
