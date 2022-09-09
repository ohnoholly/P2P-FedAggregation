from utils import *
from Core import *
import pandas as pd

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
