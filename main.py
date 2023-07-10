import pandas as pd
import numpy as np

# import data
df = pd.read_csv(r'train.csv')
dft = pd.read_csv(r'test.csv')

#read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
df_38 = df.loc[df['competition-num'] == 38]
df_39 = df.loc[df['competition-num'] == 39]
df_40 = df.loc[df['competition-num'] == 40]
df_41 = df.loc[df['competition-num'] == 41]
df_42 = df.loc[df['competition-num'] == 42]
df3 = df_41.loc[df_41['label'] == 5]
df4 = df.loc[[2]]

#K-Nearest-Neighbour main function
def KNN(df, df_s, k, n):
    knn = np.zeros(shape=(k,2))
    #print(knn)
    for i in range(0, 20000):
        if df.at[i, 'fun-rank'] == -1 & df.at[i, 'innovation-rank'] == -1 & df.at[i, 'theme-rank'] == -1 & df.at[i, 'graphics-rank'] == -1 & df.at[i, 'audio-rank'] == -1 & df.at[i, 'humor-rank'] == -1 & df.at[i, 'mood-rank'] == -1:
            continue
        sum = abs(df.at[i, 'fun-average'] - df_s.at[n, 'fun-average']) + abs(df.at[i, 'innovation-average'] - df_s.at[n, 'innovation-average']) + abs(df.at[i, 'theme-average'] - df_s.at[n, 'theme-average']) + abs(df.at[i, 'graphics-average'] - df_s.at[n, 'graphics-average']) + abs(df.at[i, 'audio-average'] - df_s.at[n, 'audio-average']) + abs(df.at[i, 'humor-average'] - df_s.at[n, 'humor-average']) + abs(df.at[i, 'mood-average'] - df_s.at[n, 'mood-average'])
        if (i < k):
            knn[i][0] = i
            knn[i][1] = sum
        else:
            knn = knn[knn[:, 1].argsort()]
            if sum < knn[k-1][1]:
                knn[k-1][0] = i
                knn[k-1][1] = sum
    vote = np.zeros(4)
    for i in range(0, k):
        j = int(knn[i][0])
        #print(j)
        if (df.at[j, 'label'] == 5):
            vote[0] = vote[0] + 1
        elif(df.at[j, 'label'] == 4):
            vote[1] = vote[1] + 1
        elif(df.at[j, 'label'] == 3):
            vote[2] = vote[2] + 1
        else:
            vote[3] = vote[3] + 1
    a = np.argmax(vote)
    if (a == 0):
        return 5
    elif(a == 1):
        return 4
    elif(a == 2):
        return 3
    else:
        return 2

#counter = 0
label = np.zeros(4959)
for i in range(0, 4959):
    if dft.at[i, 'fun-rank'] == -1 & dft.at[i, 'innovation-rank'] == -1 & dft.at[i, 'theme-rank'] == -1 & dft.at[i, 'graphics-rank'] == -1 & dft.at[i, 'audio-rank'] == -1 & dft.at[i, 'humor-rank'] == -1 & dft.at[i, 'mood-rank'] == -1:
        #counter = counter+1
        continue
    df_s = dft.loc[[i]]
    print(i)
    a = KNN(df, df_s, 3, i)
    label[i] = a
    #if (a == df.at[i, 'label']):
        #counter = counter + 1

#print(counter/2561)
np.savetxt('label.txt', label)