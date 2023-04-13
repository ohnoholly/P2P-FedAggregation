from Core import HSphereSMOTE
from Core import P2P_Aggregation
from Utils import utils
from Utils import models
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn import metrics
from torch.autograd import Variable
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim



# CLASS TO MODEL THE USERS IN THE FEDERATED LEARNING TASK
class user:
    def __init__(self,x,y,p_i,idx):
        self.xtr = x #Training data
        self.ytr = y #Labels
        self.p = p_i #Contribution to the overall model
        self.n = x.size()[0] #Number of training points provided
        self.id = idx #ID for the user
        self.v = 0.0 # the data variance on user
        self.ne_ds = 0 # The sum of data samples from all neighbors
        self.ne_dv = 0. #The sum of the data variance from all neighbors
        self.ne_sim = [] #The sum of the model similarity from all neighbors in different epochs
        self.ne_div = [] #The sum of the model divergence from all neighbors in different epochs
        self.deg = 0. #The degree of the node
        self.xtst =[] #Test data
        self.ytst =[] #Test label
        self.ne_weight = []
        self.ne_datasize = []
        self.ne_var = []
        self.ne_deg = []
        self.ne_params = {}
        self.model = Classifier_nonIID()
        self.tune_model = Classifier_nonIID()
        self.opt = []
        self.tune_opt = []
        self.loss = []
        self.tune_loss = []
        self.p_epoch = []
        self.poison_flag = False # The flag to indicate whether it is model posion
        self.adv_flag = False #The flag to indicate whether it is adversary
        self.share = 0 # the number of users this user has been shared
        self.nos = 0 # Number for sample from other clients
        self.xsyn =[]
        self.ysyn =[]
        self.o_xsyn = []
        self.o_ysyn = []


# FUNCTION TO TRAIN THE CLASSIIFER
def train_classifier(cl, opt, loss, x, y, device):
    x.to(device)
    y.to(device)
    #Reset gradients
    opt.zero_grad()
    #Train on real data
    pred = cl(x)
    pred.to(device)
    pred = torch.squeeze(pred, 1)
    err = loss(pred, y).to(device)
    err.backward()

    #Update optimizer
    opt.step()
    return err, pred

#FUNCTION FOR PREDICTION
def predict(net, Xtest, device):
    net.to(device)
    Xtest.to(device)
    with torch.no_grad():
        Ypred = net.forward(Xtest).to(device)
        label = Ypred > 0.5
    return Ypred, label


# FUNCTION TO COMPUTE THE TEST ERROR, FALSE POSITIVE AND FALSE NEGATIVE RATE, AND AUC
def computeTestErrors(cl,Xtst,Ytst, device):
    with torch.no_grad():
        err = 0

        Ypred = cl.forward(Xtst.to(device)).to(device)
        fpr, tpr, thresholds = metrics.roc_curve(Ytst.cpu().numpy(), Ypred.cpu().numpy())
        precision, recall, _ = metrics.precision_recall_curve(Ytst.cpu().numpy(), Ypred.cpu().numpy())
        #fscore = metrics.f1_score(Ytst.cpu().numpy(), Ypred.cpu().numpy())
        #plt.plot(fpr,tpr)
        #plt.show()
        #AUC
        auc = metrics.auc(fpr, tpr)
        print("AUC: ", auc)
        label = Ypred.to("cpu") > 0.5
        label = label.type(torch.float32)
        label = torch.squeeze(label, 1)

        #Classification error
        errors = 0
        for i in range(Ytst.size(0)):
            if (label[i] != Ytst[i]):
                errors += 1
        cl_error = errors/Ytst.size(0)
        #print("Classification error: ", 100*cl_error, "%")

        print(Ytst.size(0))
        print(label.shape)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(Ytst.size(0)):
            if (label[i]==1 and Ytst[i]==1):
                tp += 1

            elif (label[i]==0 and Ytst[i]==0):
                tn += 1

            elif (label[i]==1 and Ytst[i]==0):
                fp += 1

            elif (label[i]==0 and Ytst[i]==1):
                fn += 1
            else:
                print("Else!")


        print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
        false_positive_rate = fp/(Ytst.size(0) - Ytst.sum()).numpy()
        print("False positive rate: ",100*false_positive_rate, "%")

        false_negative_rate = fn/Ytst.sum().numpy()
        print("False negative rate: ",100*false_negative_rate, "%")

        p_at_r75 = precision[recall>0.75][-1]
        r_at_p75 = recall[precision>0.75][0]
        print("Precision:", p_at_r75, "Recall:", r_at_p75)
        pre = tp/ (tp+fp + 0.00000001)
        re = tp/(tp+fn)
        fscore = 2*(pre*re)/(pre+re+0.0000001)
        print("Fscore: ", fscore)


    return auc, cl_error, false_positive_rate, false_negative_rate, p_at_r75, r_at_p75, fscore

# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITHOUT REBALANCING
def train_user(cl,opt,loss,x,y,epochs,batch_size, device):
    for i in range(epochs):
            #print("Epoch user: ",i)
            #Shuffle training data
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            for beg_i in range(0, x.size(0), batch_size):
                xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(cl, opt, loss, xpo, ypo, device)
    return err, pred


#Function for upsampling from the minority class
def upSample(Xtr,Ytr,ratio):
    #Split the data according to the class
    n1 = int(Ytr.sum())
    n = int(Ytr.size(0))
    idx0 = (Ytr == 0)
    idx1 = (Ytr == 1)
    x0 = Xtr[idx0,:]
    y0 = Ytr[idx0]
    x1 = Xtr[idx1,:]
    y1 = Ytr[idx1]
    n0 = n - n1

    Xtr2 = torch.empty(0, Xtr.size(1))
    Ytr2 = torch.empty(0)
    if n0 > n1:
        #Number of samples for the minority class after upsampling
        n1_up = round(ratio*n1)

        #We sample with replacement from the minority class
        ind = torch.randint(0,n1,(n1_up,))
        x1_1 = x1[ind,:]
        y1_1 = y1[ind]

        print(x1_1.shape)
        print(x0.shape)
        #Create the new dataset
        Xtr2 = torch.cat((x0,x1_1),dim=0)
        print(Xtr2.shape)
        Ytr2 = torch.cat((y0,y1_1))
        r = torch.randperm(Xtr2.size(0))
        Xtr2 = Xtr2[r]
        Ytr2 = Ytr2[r]

    else:
        print("Upsampling n0")
        #Number of samples for the minority class after upsampling
        n0_up = round(ratio*n0)

        #We sample with replacement from the minority class
        ind = torch.randint(0,n0,(n0_up,))
        x0_0 = x0[ind,:]
        y0_0 = y0[ind]

        print(x0_0.shape)
        #Create the new dataset
        Xtr2 = torch.cat((x0_0, x1),dim=0)
        Ytr2 = torch.cat((y0_0, y1))
        r = torch.randperm(Xtr2.size(0))
        Xtr2 = Xtr2[r]
        Ytr2 = Ytr2[r]


    return Xtr2, Ytr2

#Function for downsampling
def downSample(Xtr,Ytr,ratio):
    #Split the data for the two classes
    n1 = int(Ytr.sum())
    n = int(Ytr.size(0))
    n0 = n - n1
    idx0 = (Ytr == 0)
    idx1 = (Ytr == 1)
    x0 = Xtr[idx0,:]
    y0 = Ytr[idx0]
    x1 = Xtr[idx1,:]
    y1 = Ytr[idx1]

    Xtr2 = torch.empty(0, Xtr.size(1))
    Ytr2 = torch.empty(0)
    if n0 > n1:
        n0_down = round(ratio*n0)
        #Create a new dataset with ratio*n0 samplpes for class 0
        #Note that we're shuffling the data before we call the function
        #at training time, so we don't have to shuffle here
        Xtr2 = torch.cat((x0[:n0_down],x1),dim=0)
        Ytr2 = torch.cat((y0[:n0_down],y1))
        #But we need to shuffle after we concatenate the samples
        #from the two classes
        r = torch.randperm(Xtr2.size(0))
        Xtr2 = Xtr2[r]
        Ytr2 = Ytr2[r]
    else:
        print("Downsampling n1")
        n1_down = round(ratio*n1)
        #Create a new dataset with ratio*n0 samplpes for class 0
        #Note that we're shuffling the data before we call the function
        #at training time, so we don't have to shuffle here
        Xtr2 = torch.cat((x0,x1[:n1_down]),dim=0)
        Ytr2 = torch.cat((y0,y1[:n1_down]))
        #But we need to shuffle after we concatenate the samples
        #from the two classes
        r = torch.randperm(Xtr2.size(0))
        Xtr2 = Xtr2[r]
        Ytr2 = Ytr2[r]

    return Xtr2, Ytr2


def SK_SMOTE(X, Y):
    sm = SMOTE()
    Xtr2, Ytr2 = sm.fit_resample(X, Y)


    Xtr2 = torch.from_numpy(Xtr2)
    Ytr2 = torch.from_numpy(Ytr2)
    Xtr2,Ytr2 = Xtr2.type(torch.FloatTensor),Ytr2.type(torch.FloatTensor)
    return Xtr2, Ytr2




# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH UPSAMPLING
def train_user_upsampling(u,epochs,batch_size,ratio, global_e, device):

    if global_e % 20 == 0:
        x2, y2 = upSample(u.xtr,u.ytr,ratio)
        u.xsyn = x2
        u.ysyn = y2
        u.share = False

    x = u.xsyn
    y = u.ysyn
    for i in range(epochs):
            #print("Epoch user: ",i)
            #Shuffle training data
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            for beg_i in range(0, x.size(0), batch_size):
                xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo, device)

    return err, pred

# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH DOWNSAMPLING
def train_user_downsampling(u,epochs,batch_size,ratio, global_e, device):

    if global_e % 20 == 0:
        x2, y2 = downSample(u.xtr,u.ytr,ratio)
        u.xsyn = x2
        u.ysyn = y2
        u.share = False

    x = u.xsyn
    y = u.ysyn
    for i in range(epochs):
            #print("Epoch user: ",i)
            #Shuffle training data
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            for beg_i in range(0, x.size(0), batch_size):
                xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo, device)
    return err, pred

# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH HSphereSMOTE
def train_user_HSphereSMOTE(u,epochs,batch_size,ratio, k, nusers, device):


    for i in range(epochs):
        #print("Epoch user: ",i)
        #Shuffle training data
        r = torch.randperm(u.xtr.size()[0])
        u.xtr = u.xtr[r]
        u.ytr = u.ytr[r]

        xsyn, ysyn, nls, n0, n1 = HSphereSMOTE.Sampling(u.xtr,u.ytr,ratio, k)
        u.xsyn = xsyn
        u.ysyn = ysyn
        if n0 > n1:
            u.nos = n0 - (nls+n1)
        else:
            u.nos = n1 - (nls+n0)


        x = torch.cat((u.xtr, u.xsyn), dim=0)
        y = torch.cat((u.ytr, u.ysyn), dim=0)

        if len(u.o_xsyn)!= 0:
            x = torch.cat((x, u.o_xsyn), dim=0)
            y = torch.cat((y, u.o_ysyn), dim=0)


        print(x.shape)


        for beg_i in range(0, x.size(0), batch_size):
            xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
            ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
            err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo, device)

    u.share = 0
    u.o_xsyn = []
    u.o_ysyn = []

    return err, pred



# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH SMOTE
def train_user_SMOTE(u,epochs,batch_size, device):

    for i in range(epochs):
        #print("Epoch user: ",i)
        #Shuffle training data
        r = torch.randperm(u.xtr.size()[0])
        u.xtr = u.xtr[r]
        u.ytr = u.ytr[r]
        xsyn, ysyn = SK_SMOTE(u.xtr,u.ytr)
        u.xsyn = xsyn
        u.ysyn = ysyn

        x = u.xsyn
        y = u.ysyn


        for beg_i in range(0, x.size(0), batch_size):
            xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
            ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
            err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo, device)
    return err, pred





# FUNCTION TO MERGE THE MODELS
def merge_models_ptp(u_or, u_dest, mode, nusers, comb):
    params_dest = u_dest.model.named_parameters()
    dict_params_dest = dict(params_dest)
    params_or = u_or.model.named_parameters()
    for name1, param1 in params_or:
        if name1 in dict_params_dest:
            dict_params_dest[name1].data.copy_(u_or.p*param1.data +
                            comb*dict_params_dest[name1].data)


    if len(u_or.xsyn)!=0 and u_or.share < (nusers-1) and mode ==5:
        print(u_dest.xsyn.shape)
        r = torch.randperm(round(u_dest.nos/(nusers-1)))
        o_xsample = u_or.xsyn[r]
        o_ysample = u_or.ysyn[r]
        print(o_xsample.shape)



def trainFA_imbalanced(the_model, training_data, training_labels, mode, lam, g_epochs,
                        partial_epochs, device, batch_size=128, iid=True, test_data='', test_labels='',
                        gdata='', glabel=''):

    nusers = len(training_labels)
    p0 = 1/nusers

    #TRAINING PARAMETERS (TO BE CHANGED: THIS SHOULD BE GIVEN AS A PARAMETER)
    epochs = g_epochs #TOTAL NUMBER OF TRAINING ROUNDS
    partial_epochs = partial_epochs #NUMBER OF EPOCHS RUN IN EACH CLIENT BEFORE SENDING BACK THE MODEL UPDATE
    batch_size = batch_size #BATCH SIZE


    print("Creating users...")
    users = []

    #Number of training points from class 0 for every user
    n0users = torch.zeros(nusers)
    #Number of training points from class 1 for every user
    n1users = torch.zeros(nusers)

    #CREATE MODEL
    model = the_model.to(device)

    for i in range(nusers):
        users.append(user(training_data[i],training_labels[i],p0,i+1, model))
        n1users[i] = training_labels[i].sum()
        n0users[i] = training_labels[i].size(0) - training_labels[i].sum()
        if iid == True:
            users[i].xtst = test_data[i]
            users[i].ytst = test_labels[i]


    #Total number of training samples from class 0 and 1
    N0 = n0users.sum().numpy()
    N1 = n1users.sum().numpy()

    ntr = 0
    v_sum = 0
    for u in users:
        ntr += u.xtr.size(0)
        v = utils.getvariance(u.xtr)
        u.v = torch.norm(v, float('inf'))
        print(u.v)
        v_sum += u.v

    # Weight the value of the update of each user according to the number of training data points
    # Assign model to each user
    for u in users:
        datasize = u.xtr.size(0)/ntr
        datavar = (u.v/v_sum)
        u.p = lam*datavar + (1-lam)*datasize
        print("Weight for user ", u.id, ": ", np.round(u.p,3))
        u.model = copy.deepcopy(model).to(device)
        #u.opt = optim.SGD(u.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-3)
        u.opt = optim.Adam(u.model.parameters(), lr=0.0001, weight_decay=1e-5)
        u.loss = nn.BCELoss()
        u.auc = 0.0


    for e in range(epochs):
        print("Epoch... ",e)

        #Share model with the users
        for u in range(len(users)):
            ratio = 0.9
            error, pred = train_user_HSphereSMOTE(users[u],partial_epochs,batch_size,ratio, 20, nusers, device)
            print("Loss:", error)


        if iid == True:
            auc = 0.0
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model,users[u].xtst,users[u].ytst, device)

        else:
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model, gdata, glabel, device)


        for u in users:
            comb = u.p
            for j in users:
                if j != u:
                    print("merge model", u.id)
                    merge_models_ptp(j, u, mode, nusers, comb)
                    comb = 1.0



def trainP2P(the_model, training_data, training_labels, lam, adj_list, g_epochs,
                        partial_epochs, tune_epochs, device, batch_size=128, iid=True, test_data='', test_labels='',
                        gdata='', glabel='', balanced=True):

    nusers = len(training_labels)
    p0 = 1/nusers

    #TRAINING PARAMETERS (TO BE CHANGED: THIS SHOULD BE GIVEN AS A PARAMETER)
    epochs = g_epochs #TOTAL NUMBER OF TRAINING ROUNDS
    partial_epochs = partial_epochs #NUMBER OF EPOCHS RUN IN EACH CLIENT BEFORE SENDING BACK THE MODEL UPDATE
    tune_epochs = tune_epochs
    batch_size = batch_size #BATCH SIZE
    ad_array = np.array(adj_list)
    conn = np.sum(ad_array, 1)
    deg = np.sum(ad_array)

    errorIterations = []

    print("Creating users...")
    users = []
    #TopK-Sparsification
    compressor = P2P_Aggregation.TopKCompressor()
    updater = P2P_Aggregation.model_update()

    #Number of training points from class 0 for every user
    n0users = torch.zeros(nusers)
    #Number of training points from class 1 for every user
    n1users = torch.zeros(nusers)

    for i in range(nusers):
        users.append(user(training_data[i],training_labels[i],p0,i+1))
        n1users[i] = training_labels[i].sum()
        n0users[i] = training_labels[i].size(0) - training_labels[i].sum()
        users[i].deg = conn[i]/deg
        if iid == True:
            users[i].xtst = test_data[i]
            users[i].ytst = test_labels[i]


    #Total number of training samples from class 0 and 1
    N0 = n0users.sum().numpy()
    N1 = n1users.sum().numpy()


    for u in users:
        ntr = 0 #number of all connected training set
        v_sum = 0
        ntr += u.xtr.size(0)
        v = utils.getvariance(u.xtr)
        u.v = torch.norm(v, float('inf'))
        v_sum += u.v
        ids = 0
        for ad in adj_list[u.id-1]:
            if ad == 1:
                ntr += users[ids].xtr.size(0)
                v = utils.getvariance(users[ids].xtr)
                users[ids].v = torch.norm(v, float('inf'))
                v_sum += users[ids].v

            ids = ids+1

        u.ne_ds = ntr
        u.ne_dv = v_sum




    #CREATE MODEL
    model = the_model.to(device)
    errorsEpochs = torch.zeros(epochs)

    # Weight the value of the update of each user according to the number of training data points
    # Assign model to each user
    u_p = []
    for u in users:
        ne_weight = []
        ids = 0
        for ad in adj_list[u.id-1]:
            if ad == 1:
                datasize = users[ids].xtr.size(0)/u.ne_ds
                datavar = (users[ids].v/u.ne_dv)
                #FedAvg
                p = 1/sum(adj_list[u.id-1],1)
                #p = lam*datavar + 0.5*(1-lam)*datasize + 0.5*(1-lam)*users[ids].deg
                print("Weight for user ", users[ids].id, ": ", np.round(p,3))
                ne_weight.append(p)
                u.ne_datasize.append(datasize)
                u.ne_var.append(datavar)
                u.ne_deg.append(users[ids].deg)
            else:
                if ids == u.id-1:
                    datasize = u.xtr.size(0)/u.ne_ds
                    datavar = (u.v/u.ne_dv)
                    #FedAvg
                    p = 1/sum(adj_list[u.id-1],1)
                    #p = lam*datavar + 0.5*(1-lam)*datasize + 0.5*(1-lam)*u.deg
                    print("Weight for user ", u.id, ": ", np.round(p,3))
                    ne_weight.append(p)
                    u.ne_datasize.append(datasize)
                    u.ne_var.append(datavar)
                    u.ne_deg.append(u.deg)
                else:
                    ne_weight.append(0)
                    u.ne_datasize.append(0.)
                    u.ne_var.append(0.)
                    u.ne_deg.append(0.)
            ids = ids+1

        u_p.append(ne_weight)
        u.ne_weignt = ne_weight
        u.model = copy.deepcopy(model).to(device)
        u.tune_model = copy.deepcopy(model).to(device)
        #u.opt = optim.SGD(u.model.parameters(), lr=0.00001, momentum=0.9, weight_decay=10)
        u.opt = optim.Adam(u.model.parameters(), lr=0.0001)
        u.tune_opt = optim.Adam(u.model.parameters(), lr=0.0001)
        u.loss = nn.BCELoss()
        u.tune_loss = nn.BCELoss()

    var = []
    for u in users:
        var.append(u.v.item())


    sort_var = sorted(var, key=float, reverse=True)

    max_id = var.index(sort_var[0])
    second = var.index(sort_var[1])
    third = var.index(sort_var[2])
    print(max_id)
    print(second)
    print(third)

    pre_e = 1
    for e in range(epochs):
        print("Epoch... ",e)
        epoch_i_array.append(e)

        if balanced == True:
            #Share model with the users
            for u in range(len(users)):
                print(users[u].adv_flag)
                start = time.time()
                error, pred = train_user(users[u].model,users[u].opt,users[u].loss,users[u].xtr,users[u].ytr,partial_epochs,batch_size, users[u].poison_flag)
                end = time.time()
                local_training_time = end-start
                print("training_time:", local_training_time)
                print("Loss:", error)
        else:
            #Share model with the users
            for u in range(len(users)):
                print(users[u].adv_flag)
                ratio = 0.9
                error, pred = train_user_HSphereSMOTE(users[u],partial_epochs,batch_size,ratio, 20, e, nusers, device)



        #Compute test accuracy

        if iid == True:
            auc = 0.0
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model,users[u].xtst,users[u].ytst)


        else:
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model, gdata, glabel)

        """
        Create the mask for random-K sparsification.
        """
        #Create the mask for sparsification (rand and topk)
        masks_dict = {}
        for name, param in model.named_parameters():
            size = list(param.shape)
            mask = masks(size, 50).to(device)
            masks_dict[name] = mask


        """
        Launch the attacks
        """
        #Launch attack in epoch 50

        #if e == 0:
            #Label flipping attack
            #fliped_a = label_flipping(users[max_id], 0.95)
            #fliped_b = label_flipping(users[second], 0.95)
            #fliped_c = label_flipping(users[third], 0.95)
            #users[max_id].ytr.copy_(fliped_a)
            #users[second].ytr.copy_(fliped_b)
            #users[third].ytr.copy_(fliped_c)

            #Noise attack
            #noised_xtr, noised_ytr = adding_noise(users[max_id], 2)
            #noised_xsec, noised_ysec = adding_noise(users[second], 2)
            #noised_xthi, noised_ythi = adding_noise(users[third], 2)

            #users[max_id].xtr = noised_xtr
            #users[max_id].ytr = noised_ytr
            #users[second].xtr = noised_xsec
            #users[second].ytr = noised_ysec
            #users[third].xtr = noised_xthi
            #users[third].ytr = noised_ythi

            #Model poisoning
            #users[max_id].poison_flag = True
            #users[second].poison_flag = True
            #users[third].poison_flag = True

            #users[max_id].adv_flag = True
            #users[second].adv_flag = True
            #users[third].adv_flag = True

        #Byazantine attack
        if e>=0:
            parameter_poison(users[max_id])
            #parameter_poison(users[second])
            #parameter_poison(users[third])
            if users[max_id].adv_flag == False:
                users[max_id].adv_flag = True
                #users[second].adv_flag = True
                #users[third].adv_flag = True


        shared = False
        """
        Model aggregation and updating
        """

        merge_times = []
        for u in range(len(users)):
            weights = u_p[u]
            print(weights)
            comb = weights[u] # the weight for the current node

            sim_sum = 0
            div_sum = 0
            ids = 0

            for ad in adj_list[u]:
                if ad == 1:
                    print("merge model", users[ids].id)
                    updater.sparse_merge_top(users[ids], users[u], comb, weights[ids], adj_list, compressor, shared)

                    sim_sum = sim_sum + updater.similarity[users[u].id][users[ids].id][e]
                    div_sum = div_sum + updater.divergence[users[u].id][users[ids].id][e]
                    comb = 1.0

                ids = ids + 1
            users[u].ne_params = updater.params[users[u].id]
            end_merge_time = time.time()
            users[u].ne_sim.append(sim_sum+1)  #+1 is for the similarity to its own model
            users[u].ne_div.append(div_sum)

        """
        Tunning of the weight of node
        """

        if e == 1 or (e%10==0 and e!=0):
            start_tune_time = time.time()
            for u in users:
                if u.adv_flag == False:
                    ww, penal = Bayes_Optimizer(u, adj_list, updater, tune_epochs, batch_size, e)
                    print("ww:",ww)
                    print("Penal:",penal)

                    penalty = 0.
                    for ids, ad in enumerate(adj_list[u.id-1]):
                        #If the node is adjecent
                        if ad == 1:
                            pens = penal *(updater.divergence[u.id][ids+1][e]/u.ne_div[e])
                            score = ww[0]*u.ne_datasize[ids]+ww[1]*u.ne_var[ids]+ww[2]*u.ne_deg[ids]+ww[3]*(updater.similarity[u.id][ids+1][e]/u.ne_sim[e])-pens
                            print("Weight for user ", users[ids].id, ": ", np.round(score,3))
                            u_p[u.id-1][ids] = score
                            penalty = penalty + pens


                    u_weight = ww[0]*u.ne_datasize[u.id-1]+ww[1]*u.ne_var[u.id-1]+ww[2]*u.ne_deg[u.id-1]+ww[3]*(1/u.ne_sim[e])+penalty
                    print("Weight for user ", u.id, ": ", np.round(u_weight,3))
                    u_p[u.id-1][u.id-1] = u_weight
                    u_p[u.id-1] = [0 if i < 0 else i for i in u_p[u.id-1]]
                    print(u_p[u.id-1])
                    num_p = sum(i > 0 for i in u_p[u.id-1])
                    print(num_p)
                    thres = sum(u_p[u.id-1])/num_p
                    print(thres)
                    if u_p[u.id-1][u.id-1]>thres:
                        add = u_p[u.id-1][u.id-1]-thres
                        add = add/(num_p-1)
                        print(add)
                        u_p[u.id-1][u.id-1] = thres
                        for it, i in enumerate(u_p[u.id-1]):
                            if i!=0:
                                if it!=u.id-1:
                                    u_p[u.id-1][it] = u_p[u.id-1][it]+add

                    print(u_p[u.id-1])
                    u_p[u.id-1] = [float(i)/sum(u_p[u.id-1]) for i in u_p[u.id-1]]
                    print(u_p[u.id-1])

            end_tune_time = time.time()
            tune_time = end_tune_time - start_tune_time
            print("Tune_time:", tune_time)

            pre_e = e


    return errorsEpochs
