#Correlation Matrix
#Script for generating correlation matrix for CG dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = 'Phish_Dataset.csv'
df = pd.read_csv(file)
add_df = df.set_index('id')
add_df = add_df[['senders_email_tld']].applymap(lambda x: x.split('.')[0])
df = df.drop(['senders_email_tld'], axis=1)
df = df.set_index('id').join(add_df)

#dfY = df[df['target_phishYN'] == 'Y']
#dfN = df[df['target_phishYN'] == 'N']

df = df.replace({'target_phishYN':{'Y':1, 'N':0}})

corr = df.corr()
#corrY = dfY.corr()
#corrN = dfN.corr()

print corr
