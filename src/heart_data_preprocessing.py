import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../data/heart.csv")

# one-hot encoding:  'cp', 'restecg', 'slope', 'ca'?, 'thal'
# onehot_vars = ['cp', 'restecg', 'slope', 'ca', 'thal']
onehot_vars = ['cp', 'restecg', 'slope', 'thal']
for var in onehot_vars:
    df_add = pd.get_dummies(df[[var]].astype(str),prefix=[var], drop_first=True)
    df = pd.concat([df, df_add], axis=1)

df.drop(onehot_vars, axis=1, inplace=True)

# move target to the last column
df_target = df.pop('target') 
df['target'] = df_target

# Scale the features
print(df.head())

scaler_x = MinMaxScaler()
df.iloc[:,:-1] = scaler_x.fit_transform(df.iloc[:,:-1])
print(df.head())

#print(scaler_x.inverse_transform(df.iloc[:,:-1]))
# save the scaler
pickle.dump(scaler_x, open("../data/heart_scaler.sav", 'wb'))

# save the scaled data
# None None means: no Dimension Reduction, and no Clustering so far
df.to_csv("../data/heart_None_None.csv", index = False)
