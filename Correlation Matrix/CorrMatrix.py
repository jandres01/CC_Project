#Script for generating correlation matrix for CG dataset.

from string import ascii_letters
from pylab import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

file = '../Phish_Dataset.csv'
df = pd.read_csv(file)
add_df = df.set_index('id')
add_df = add_df[['senders_email_tld']].applymap(lambda x: x.split('.')[0])
df = df.drop(['senders_email_tld'], axis=1)
df = df.set_index('id').join(add_df)

df = df.replace({'target_phishYN':{'Y':1, 'N':0}})

dfY = df[df['target_phishYN'] == 1]
dfN = df[df['target_phishYN'] == 0]

corrM = df.corr()
corrY = dfY.corr()
corrN = dfN.corr()

print corrM

def plotFigure(corr):
  sns.set(style="white")
  mask = np.zeros_like(corr, dtype=np.bool)
  mask[np.triu_indices_from(mask)] = True
  f, ax = plt.subplots(figsize=(9, 7))
  cmap = sns.diverging_palette(220, 10, as_cmap=True)

  # Draw the heatmap with the mask and correct aspect ratio
  fig = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

  lbls = ['phish','domID','twoDo','cFull','cUser','hasNum','hasPun','eHasA','numRe','numSu','hasSu','AttMs','attIc','attIm','attTx','attZi','attOt']

  fig.set_xticklabels(lbls,rotation=30)
  fig.set_yticklabels(lbls,rotation=0)
  return fig

fig = plotFigure(corrM)
fig.set_title('Correlation Matrix Whole Data')

lblE = ['domID','twoDo','cFull','cUser','hasNum','hasPun','eHasA','numRe','numSu','hasSu','AttMs','attIc','attIm','attTx','attZi','attOt']

figY = plotFigure(corrY)
figY.set_title('Correlation Matrix Phish Data')
figY.set_xticklabels(lblE,rotation=30)
figY.set_yticklabels(lblE,rotation=0)

figN = plotFigure(corrN)
figN.set_title('Correlation Matrix Reg Email Data')
figN.set_xticklabels(lblE,rotation=30)
figN.set_yticklabels(lblE,rotation=0)

plt.show()



