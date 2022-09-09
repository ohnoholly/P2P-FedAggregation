import torch.nn as nn

class Classifier_nonIID(nn.Module):
    def __init__(self):
        super(Classifier_nonIID,self).__init__()
        self.fc1 = nn.Linear(115, 80)
        self.bn1 = nn.BatchNorm1d(80)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(80, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(50, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.drop3 = nn.Dropout(p=0.5)
        self.out = nn.Linear(10, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class Classifier_nonIIDUNSW(nn.Module):
    def __init__(self):
        super(Classifier_nonIIDUNSW,self).__init__()
        self.fc1 = nn.Linear(9,  20)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(20, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.out = nn.Linear(10, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class Classifier_IID(nn.Module):
    def __init__(self):
        super(Classifier_IID,self).__init__()
        self.fc1 = nn.Linear(9, 20)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.out = nn.Linear(10, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        x = self.out(x)
        x = self.out_act(x)
        return x
