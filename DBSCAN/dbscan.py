# DBSCAN for CC project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#Read data & convert Y-1, N-0
def read_data():
    file = 'Phish_Dataset.csv'
    df = pd.read_csv(file)
    df = df.replace({'target_phishYN':{'Y':1, 'N':0}})
    #fix domain names
    #df = df[['fix_senders_email_tld']].applymap(lambda x: str(x).split(r'.')[0])
    return df

def main():
  df = read_data()
  mapping = {'set': 1, 'test': 2}
  df = df[['domain_SenderID','email_has_attachment','num_attachment_zip','fix_senders_email_tld']]
  
  data = df.values
  data = np.float32(data)
  max_distance = 1
  print "enter"
  db = DBSCAN(eps=max_distance, min_samples=10).fit(data)
  print "fail"
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  unique_labels = set(labels)

  # Plot up the results!
  min_x = np.min(data[:, 0])
  max_x = np.max(data[:, 0])
  min_y = np.min(data[:, 1])
  max_y = np.max(data[:, 1])

  fig = plt.figure(figsize=(12,6))
  plt.subplot(121)
  plt.plot(data[:,0], data[:,1], 'ko')
  plt.xlim(min_x, max_x)
  plt.ylim(min_y, max_y)
  plt.title('Original Data', fontsize = 20)
  print "title" 
  plt.subplot(122)
  colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
  for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)
    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)
  plt.xlim(min_x, max_x)
  plt.ylim(min_y, max_y)
  plt.title('DBSCAN: %d clusters found' % n_clusters, fontsize = 20)
  fig.tight_layout()
  plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.05)
  print("not showing")
  plt.show()
  print("fail")


if __name__ == "__main__":
  main()



#  dataY = df.pivot(index='id',columns='domain_SenderID',values='target_phishYN')
#  dataX = np.empty((107748,756,3))
#  dataX[:,:,0] = pd.pivot_table(df,values='target_phishYN',index='id',columns='domain_SenderID')
#  dataX[:,:,1] = pd.pivot_table(df,values='num_attachment_zip',index='id',columns='domain_SenderID')                 
#  dataX[:,:,2] = pd.pivot_table(df,values='email_has_attachment',index='id',columns='domain_SenderID')                 
#  y = dataY.values
#  x = dataX

