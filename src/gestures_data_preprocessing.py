import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


dir_in = '../data/emg-4/'

col_names = []
for i in range(8):
    for j in range(8):
        col_names.append('muscle reading '+ str(i+1)+' sensor '+str(j+1))
col_names.append('gestures')

# Gesture classes were : rock - 0, scissors - 1, paper - 2, ok - 3

df = []        
for i in range(4):
    fn = dir_in + str(i) + '.csv'
    df_i = pd.read_csv(fn, header=None,index_col = False, names = col_names)
    print(len(df_i))
    if len(df) == 0:
        df = df_i
    else:
        df = df.append(df_i)
print(len(df))

# Scale the features
print(df.head())

scaler_x = MinMaxScaler()
df.iloc[:,:-1] = scaler_x.fit_transform(df.iloc[:,:-1])
print(df.head())

#print(scaler_x.inverse_transform(df.iloc[:,:-1]))
# save the scaler
pickle.dump(scaler_x, open("../data/gestures_scaler.sav", 'wb'))

# save the scaled data
# None None means: no Dimension Reduction, and no Clustering so far
df.to_csv("../data/gestures_None_None.csv", index = False)
