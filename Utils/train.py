from Core import HSphereSMOTE
from Utils import utils
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from torch.autograd import Variable
import torch
import copy
import torch.nn as nn
import torch.optim as optim


# FUNCTION TO TRAIN THE CLASSIIFER
def train_classifier(cl, opt, loss, x, y):
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
def predict(net, Xtest):
    net.to(device)
    Xtest.to(device)
    with torch.no_grad():
        Ypred = net.forward(Xtest).to(device)
        label = Ypred > 0.5
    return Ypred, label


# FUNCTION TO COMPUTE THE TEST ERROR, FALSE POSITIVE AND FALSE NEGATIVE RATE, AND AUC
def computeTestErrors(cl,Xtst,Ytst):
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

# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER
def train_user(cl,opt,loss,x,y,epochs,batch_size):
    for i in range(epochs):
            #print("Epoch user: ",i)
            #Shuffle training data
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            for beg_i in range(0, x.size(0), batch_size):
                xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(cl, opt, loss, xpo, ypo)
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

def SK_ADASYN(X, Y):
    sm = ADASYN()
    Xtr2, Ytr2 = sm.fit_sample(X, Y)

    Xtr2 = torch.from_numpy(Xtr2)
    Ytr2 = torch.from_numpy(Ytr2)
    return Xtr2, Ytr2





# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH UPSAMPLING
def train_user_upsampling(u,epochs,batch_size,ratio, global_e):

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
                err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo)

    return err, pred

# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH DOWNSAMPLING
def train_user_downsampling(u,epochs,batch_size,ratio, global_e):

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
                err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo)
    return err, pred


def train_user_HSphereSMOTE(u,epochs,batch_size,ratio, k, global_e, nusers):


    for i in range(epochs):
        #print("Epoch user: ",i)
        #Shuffle training data
        r = torch.randperm(u.xtr.size()[0])
        u.xtr = u.xtr[r]
        u.ytr = u.ytr[r]

        xsyn, ysyn, nls, n0, n1 = FarK_Sampleing(u.xtr,u.ytr,ratio, k)
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
            err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo)

    u.share = 0
    u.o_xsyn = []
    u.o_ysyn = []

    return err, pred



# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH UPSAMPLING
def train_user_SMOTE(u,epochs,batch_size, global_e):

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


        #x2, y2 = SK_SMOTE(x,y)
        for beg_i in range(0, x.size(0), batch_size):
            xpo = Variable(x[beg_i:beg_i + batch_size, :]).to(device)
            ypo = Variable(y[beg_i:beg_i + batch_size]).to(device)
            err, pred = train_classifier(u.model, u.opt, u.loss, xpo, ypo)
    return err, pred


# FUNCTION TO TRAIN THE MODEL FOR A SPECIFIC USER WITH UPSAMPLING
def train_user_ADASYN(cl,opt,loss,x,y,epochs,batch_size,ratio):
    for i in range(epochs):
            #print("Epoch user: ",i)
            #Shuffle training data
            r = torch.randperm(x.size()[0])
            x = x[r]
            y = y[r]
            x2, y2 = SK_ADASYN(x,y)
            for beg_i in range(0, x2.size(0), batch_size):
                xpo = Variable(x2[beg_i:beg_i + batch_size, :]).to(device)
                ypo = Variable(y2[beg_i:beg_i + batch_size]).to(device)
                err, pred = train_classifier(cl, opt, loss, xpo, ypo)
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



def trainFA_imbalanced(training_data, training_labels, mode, lam, g_epochs,
                        partial_epochs, batch_size=128, iid=True, test_data='', test_labels='',
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

    for i in range(nusers):
        users.append(user(training_data[i],training_labels[i],p0,i+1))
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


    #CREATE MODEL
    model = Classifier_nonIID().to(device)
    errorsEpochs = torch.zeros(epochs)

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
        epoch_i_array.append(e)

        #Share model with the users
        for u in range(len(users)):

            if (mode == 1):
                #DOWNSAMPLE
                ratio = float(N0/N1)
                error, pred = train_user_downsampling(users[u],partial_epochs,batch_size,ratio,e)

            if (mode == 2):
                #UPSAMPLE
                ratio = float(N1/N0)
                error, pred = train_user_upsampling(users[u],partial_epochs,batch_size,ratio, e)

            if (mode == 3):
                #SMOTE
                ratio = float(N0/N1)
                error, pred = train_user_SMOTE(users[u],partial_epochs,batch_size, e)
                errlist[u].append(error)

            if (mode == 4):
                #UPSAMPLE
                ratio = float(N0/N1)
                error, pred = train_user_ADASYN(u.model,u.opt,u.loss,
                                                      u.xtr,u.ytr,partial_epochs,batch_size,ratio)

            if (mode == 5):
                #UPSAMPLE_FarK
                #print("N0:", N0)
                #print("N1:", N1)
                ratio = 0.9
                #print(ratio)
                error, pred = train_user_FarK(users[u],partial_epochs,batch_size,ratio, 20, e, nusers)
                print("Loss:", error)
                errlist[u].append(error)

            if (mode == 6):
                #without rebalanced peer-to-peer
                error, pred = train_user(users[u].model,users[u].opt,users[u].loss,users[u].xtr,users[u].ytr,partial_epochs,batch_size)
                print("Loss:", error)
                errlist[u].append(error)



        #Compute test accuracy

        if iid == True:
            auc = 0.0
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model,users[u].xtst,users[u].ytst)
                auclist[u].append(auc_u)
                fslist[u].append(fs)
                fplist[u].append(fpr)
                fnlist[u].append(fnr)
                parlist[u].append(par)
                raplist[u].append(rap)

        else:
            for u in range(len(users)):
                print("User:", u)
                auc_u, cl_error, fpr, fnr, par, rap, fs = computeTestErrors(users[u].model, gdata, glabel)
                auclist[u].append(auc_u)
                fslist[u].append(fs)
                fplist[u].append(fpr)
                fnlist[u].append(fnr)
                parlist[u].append(par)
                raplist[u].append(rap)


        #update_id = e % len(users)
        for u in users:
            comb = u.p
            for j in users:
                if j != u:
                    print("merge model", u.id)
                    merge_models_ptp(j, u, mode, nusers, comb)
                    comb = 1.0

    return errorsEpochs
