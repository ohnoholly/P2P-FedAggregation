import pandas as pd
import pickle

DATA_PATH = "./"

df_B1 = pd.read_csv(DATA_PATH+"/PT_838_Security Camera/benign_traffic.csv")
df_A1 = pd.read_csv(DATA_PATH+"/PT_838_Security Camera/Mirai/ack.csv")
df_B2 = pd.read_csv(DATA_PATH+"/XCS7_1002_WHT_Security_Camera/benign_traffic.csv")
df_A2 = pd.read_csv(DATA_PATH+"/XCS7_1002_WHT_Security_Camera/Mirai/ack.csv")
df_B3 = pd.read_csv(DATA_PATH+"/Danmini_Doorbell/benign_traffic.csv")
df_A3 = pd.read_csv(DATA_PATH+"/Danmini_Doorbell/Mirai/ack.csv")

#Sample the data according to its portion
total = len(df_A1)
print(total)
part_samp = int(0.2*total)
print(part_samp)
df_A1 = df_A1.iloc[0:part_samp, 0:115]


#Sample the data according to its portion
total = len(df_A2)
print(total)
part_samp = int(0.2*total)
print(part_samp)
df_A2 = df_A2.iloc[0:part_samp, 0:115]

#Sample the data according to its portion
total = len(df_A3)
print(total)
part_samp = int(0.2*total)
print(part_samp)
df_A3 = df_A3.iloc[0:part_samp, 0:115]


#Feed to FL
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
with open("./N_BaIoT_PrivateA_features_a.p", "wb") as handle:
    pickle.dump(x_a_private, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open("./N_BaIoT_PrivateA_labels_a.p", "wb") as handle:
    pickle.dump(y_a_private, handle, protocol = pickle.HIGHEST_PROTOCOL)
