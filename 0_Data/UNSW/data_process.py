import pandas as pd
import torch
import numpy as np
import pickle

DATA_PATH = "./"
df = pd.read_csv(DATA_PATH+"UNSW_2018_IoT_Botnet_Final_10_Best.csv",header=None, sep=';')
df = df.dropna()


A1 = df[df.iloc[:,18].str.match('Reconnaissance')]
A2 = df[df.iloc[:,18].str.match('DoS')]
A3 = df[df.iloc[:,18].str.match('DDoS')]
B = df[df.iloc[:,18].str.match('Normal')]

total = len(A1)
part_samp = int(0.5*total)
df_A1 = A1.iloc[0:part_samp, 7:17]

total = len(A2)
part_samp = int(0.05*total)
df_A2 = A2.iloc[0:part_samp, 7:17]

total = len(A3)
part_samp = int(0.05*total)
df_A3 = A3.iloc[0:part_samp, 7:17]

total = len(B)
part_samp = round(total/3)
iid = 0
df_B1 = B.iloc[0:part_samp, 7:17]
iid = iid + part_samp
df_B2 = B.iloc[iid:iid+part_samp, 7:17]
iid = iid + part_samp
df_B3 = B.iloc[iid:iid+part_samp, 7:17]
print(df_B1.shape)
print(df_B2.shape)
print(df_B3.shape)


tB1 = torch.FloatTensor(df_B1.values.astype(np.float32))
tA1 = torch.FloatTensor(df_A1.values.astype(np.float32))
tB2 = torch.FloatTensor(df_B2.values.astype(np.float32))
tA2 = torch.FloatTensor(df_A2.values.astype(np.float32))
tB3 = torch.FloatTensor(df_B3.values.astype(np.float32))
tA3 = torch.FloatTensor(df_A3.values.astype(np.float32))
print(tB1.shape)
print(tA1.shape)
print(tB2.shape)
print(tA2.shape)
print(tB3.shape)
print(tA3.shape)
d0 = [tB1, tB2, tB3]
d1 = [tA1, tA2, tA3]


#Creare Pickle files for the dataset being used
with open("./d0.p", "wb") as handle:
    pickle.dump(d0, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open("./d1.p", "wb") as handle:
    pickle.dump(d1, handle, protocol = pickle.HIGHEST_PROTOCOL)
