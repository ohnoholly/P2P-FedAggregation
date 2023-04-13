import torch


def getData(frac0,frac1,frtst0,frtst1,frac_tr0,frac_tr1,frac_tst0,frac_tst1,d0='', d1=''):

    #frac0 is the fraction of the benign traffc, also includes the number of user

    #Ensure that the tensor with the fraction of points (for both class 0 and 1) add up to 1
    print("frac0_before",frac0)
    frac0 = frac0/frac0.sum()
    frac1 = frac1/frac1.sum()
    print("frac0_after",frac0)

    #Ensure that the tensor with the fraction of points (for both class 0 and 1) add up to 1
    print("frac0_before",frtst0)
    frtst0 = frtst0/frtst0.sum()
    frtst1 = frtst1/frtst1.sum()
    print("frac0_after",frtst0)

    #Number of participants (users) in the federated learning task
    nusers = frac0.size(0)
    print("nusers",nusers)

    #Structures to store the data for the participants
    training_data = []
    training_labels = []
    test_data = []
    test_labels = []

    #Load the data


    #d0 = torch.load('Dataset/BNG dataset/x0.pt')
    #d1 = torch.load('Dataset/BNG dataset/x1.pt')

    x0 = d0.type(torch.float)
    x1 = d1.type(torch.float)

    print(x0.shape)
    print(x1.shape)


    #Shuffle the training data
    r = torch.randperm(x0.size(0))
    x0 = x0[r,:]
    r = torch.randperm(x1.size(0))
    x1 = x1[r,:]

    #Number of data points for each class
    n0 = x0.size(0)
    n1 = x1.size(0)

    #Number of training data points for each class
    ntr0 = round(frac_tr0*n0)
    #ntr0 = ntr0.numpy()
    ntr1 = round(frac_tr1*n1)
    #ntr1 = ntr1.numpy()

    #Number of testing data points for each class
    ntst0 = round(frac_tst0*n0)
    ntst1 = round(frac_tst1*n1)

    print("ntr0",ntr0)
    print("ntr1",ntr1)
    print("ntst0",ntst0)
    print("ntst1",ntst1)


    #Create the labels tensors
    y0 = torch.zeros(x0.size(0))
    y1 = torch.ones(x1.size(0))

    #Number of training points (from both class 0 and 1) for each user
    ntr0_users = (frac0*ntr0).floor()
    ntr1_users = (frac1*ntr1).floor()

    print("ntr0_users",ntr0_users)
    print("ntr1_users",ntr1_users)

    #Number of testing points (from both class 0 and 1) for each user
    ntst0_users = (frtst0*ntst0).floor()
    ntst1_users = (frtst1*ntst1).floor()

    print("ntst0_users",ntst0_users)
    print("ntst1_users",ntst1_users)

    it0 = 0
    it1 = 0
    #Split the training data across the different users
    for i in range(nusers):
        xx0 = x0[it0:it0+int(ntr0_users[i]),:].clone().detach()
        xx1 = x1[it1:it1+int(ntr1_users[i]),:].clone().detach()
        yy0 = y0[it0:it0+int(ntr0_users[i])].clone().detach()
        yy1 = y1[it1:it1+int(ntr1_users[i])].clone().detach()

        x = torch.cat((xx0,xx1),dim=0)
        y = torch.cat((yy0,yy1))
        r = torch.randperm(x.size(0))
        x = x[r,:]
        y = y[r]

        training_data.append(x)
        training_labels.append(y)

        it0 += int(ntr0_users[i])
        it1 += int(ntr1_users[i])

        x0test = x0[it0:it0+int(ntst0_users[i]),:].clone().detach()
        x1test = x1[it1:it1+int(ntst1_users[i]),:].clone().detach()
        y0test = y0[it0:it0+int(ntst0_users[i])].clone().detach()
        y1test = y1[it1:it1+int(ntst1_users[i])].clone().detach()

        xtest = torch.cat((x0test,x1test),dim=0)
        ytest = torch.cat((y0test,y1test))
        r = torch.randperm(xtest.size(0))
        xtest = xtest[r,:]
        ytest = ytest[r]

        test_data.append(xtest)
        test_labels.append(ytest)

        it0 += int(ntst0_users[i])
        it1 += int(ntst1_users[i])


    return training_data,training_labels,test_data,test_labels


# Function to normalize the data in a federated way,
# i.e. considering the mean and standard deviation of
# each shard

def normalizeData(training_data, test_data):
    #Number of users
    nusers = len(training_data)
    #Number of features
    d = test_data[0].size(1)

    m_tr = torch.zeros(nusers,d,dtype=torch.float)
    frac_tr = torch.zeros(nusers,d,dtype=torch.float)

    m_tst = torch.zeros(nusers,d,dtype=torch.float)
    frac_tst = torch.zeros(nusers,d,dtype=torch.float)

    #Get mean from training data
    it = 0
    for x in training_data:
        frac_tr[it,:] = x.size(0)
        m_tr[it,:] = x.mean(dim=0)
        it += 1

    frac_tr = frac_tr/frac_tr.sum(dim=0)
    mm_tr = (m_tr * frac_tr).sum(dim=0)

    #Get mean from test data
    it = 0
    for x in test_data:
        frac_tst[it,:] = x.size(0)
        m_tst[it,:] = x.mean(dim=0)
        it += 1

    frac_tst = frac_tst/frac_tst.sum(dim=0)
    mm_tst = (m_tst * frac_tst).sum(dim=0)

    #Get std from training data
    training_data2 = []
    s_tr = torch.zeros(nusers,d,dtype=torch.float)
    it = 0
    for x in training_data:
        x2 = (x - mm_tr)
        s_tr[it,:] = x2.std(dim=0)
        training_data2.append(x2)
        #print("Mean here: ", x2.mean(dim=0))
        it += 1

    ss_tr = (s_tr * frac_tr).sum(dim=0)

    #Get std from training data
    test_data2 = []
    s_tst = torch.zeros(nusers,d,dtype=torch.float)
    it = 0
    for x in test_data:
        x2 = (x - mm_tst)
        s_tst[it,:] = x2.std(dim=0)
        test_data2.append(x2)
        #print("Mean here: ", x2.mean(dim=0))
        it += 1

    ss_tst = (s_tst * frac_tst).sum(dim=0)


    del training_data
    del test_data

    training_data = []
    for x in training_data2:
        x2 = x/ss_tr
        training_data.append(x2)

    test_data = []
    for x in test_data2:
        x2 = x/ss_tst
        test_data.append(x2)


    return training_data, test_data


# FUNCTION TO SPLIT THE DATA ACROSS THE DIFFERENT USERS IN THE LEARNING TASK

def getData_nonIID(frac0,frac1, frtst0,frtst1, frac_tr0,frac_tr1, frac_tst0,frac_tst1, d0='', d1=''):

    #frac0 is the fraction of the benign traffc, also includes the number of user


    #Number of participants (users) in the federated learning task
    nusers = frac0.size(0)
    print("nusers",nusers)

    #Structures to store the data for the participants
    training_data = []
    training_labels = []
    global_valdata = []
    global_labels = []

    #Load the data

    for i in range(nusers):

        x0 = d0[i].type(torch.float)
        x1 = d1[i].type(torch.float)

        print(x0.shape)
        print(x1.shape)


        #Shuffle the training data
        r = torch.randperm(x0.size(0))
        x0 = x0[r,:]
        r = torch.randperm(x1.size(0))
        x1 = x1[r,:]

        #Number of data points for each class
        n0 = x0.size(0)
        n1 = x1.size(0)

        #Number of training data points for each class
        ntr0 = round(frac_tr0*n0)
        #ntr0 = ntr0.numpy()
        ntr1 = round(frac_tr1*n1)
        #ntr1 = ntr1.numpy()

        #Number of testing data points for each class
        ntst0 = round(frac_tst0*n0)
        ntst1 = round(frac_tst1*n1)

        print("ntr0",ntr0)
        print("ntr1",ntr1)
        print("ntst0",ntst0)
        print("ntst1",ntst1)


        #Create the labels tensors
        y0 = torch.zeros(x0.size(0))
        y1 = torch.ones(x1.size(0))

        #Number of training points (from both class 0 and 1) for each user
        ntr0_users = (frac0[i]*ntr0).floor()
        ntr1_users = (frac1[i]*ntr1).floor()

        print("ntr0_users",ntr0_users)
        print("ntr1_users",ntr1_users)

        #Number of global testing points (from both class 0 and 1) for each user
        nglobalva0 = (frtst0[i]*(ntst0)).floor()
        nglobalva1 = (frtst1[i]*(ntst1)).floor()

        print("nglobalva0",nglobalva0)
        print("nglobalva1",nglobalva1)

        it0 = 0
        it1 = 0

        xx0 = x0[it0:it0+int(ntr0_users),:].clone().detach()
        xx1 = x1[it1:it1+int(ntr1_users),:].clone().detach()
        yy0 = y0[it0:it0+int(ntr0_users)].clone().detach()
        yy1 = y1[it1:it1+int(ntr1_users)].clone().detach()

        x = torch.cat((xx0,xx1),dim=0)
        y = torch.cat((yy0,yy1))
        r = torch.randperm(x.size(0))
        x = x[r,:]
        y = y[r]

        training_data.append(x)
        training_labels.append(y)

        it0 += int(ntr0_users)
        it1 += int(ntr1_users)

        x0glo = x0[it0:it0+int(nglobalva0),:].clone().detach()
        x1glo = x1[it1:it1+int(nglobalva1),:].clone().detach()
        y0glo = y0[it0:it0+int(nglobalva0)].clone().detach()
        y1glo = y1[it1:it1+int(nglobalva1)].clone().detach()

        xglo = torch.cat((x0glo,x1glo),dim=0)
        yglo = torch.cat((y0glo,y1glo))
        r = torch.randperm(xglo.size(0))
        xglo = xglo[r,:]
        yglo = yglo[r]

        global_valdata.append(xglo)
        global_labels.append(yglo)


    global_testdata = global_valdata[0]
    global_testlabels = global_labels[0]
    for u in range(len(global_valdata)-1):
        global_testdata = torch.cat((global_testdata, global_valdata[u+1]),dim=0)
        global_testlabels = torch.cat((global_testlabels, global_labels[u+1]))

    print(global_testdata.shape)
    print(global_testlabels.shape)

    return training_data,training_labels, global_testdata, global_testlabels

def normalizeData_nonIID(training_data):
    #Number of users
    nusers = len(training_data)
    #Number of features
    d = training_data[0].size(1)

    m_tr = torch.zeros(nusers,d,dtype=torch.float)
    frac_tr = torch.zeros(nusers,d,dtype=torch.float)

    #Get mean from training data
    it = 0
    for x in training_data:
        frac_tr[it,:] = x.size(0)
        m_tr[it,:] = x.mean(dim=0)
        it += 1

    frac_tr = frac_tr/frac_tr.sum(dim=0)
    mm_tr = (m_tr * frac_tr).sum(dim=0)


    #Get std from training data
    training_data2 = []
    s_tr = torch.zeros(nusers,d,dtype=torch.float)
    it = 0
    for x in training_data:
        x2 = (x - mm_tr)
        s_tr[it,:] = x2.std(dim=0)
        training_data2.append(x2)
        #print("Mean here: ", x2.mean(dim=0))
        it += 1

    ss_tr = (s_tr * frac_tr).sum(dim=0)

    del training_data


    training_data = []
    for x in training_data2:
        x2 = x/ss_tr
        training_data.append(x2)


    return training_data
