#Correlation Matrix
#Script for generating correlation matrix for CG dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'Phish_Dataset.csv'
df = pd.read_csv(file)

dfY = df[df['target_phishYN']=="Y"].drop(['id'], axis=1)
dfN = df[df['target_phishYN']=="N"].drop(['id'], axis=1)

corrY = dfY.corr()
corrN = dfN.corr()

