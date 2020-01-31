import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


sperm_data = pd.read_csv(r"all_features.csv")
sperm_data = sperm_data.replace("-", np.nan)
print(sperm_data.shape)
print(type(sperm_data))
sperm_data = sperm_data.dropna()
print(sperm_data.shape)
print(type(sperm_data))

# Split data to train and test
train_features, test_features, train_blast, test_blast = train_test_split(
    sperm_data.drop(labels=['BLAST_D8'], axis=1),
    sperm_data['BLAST_D8'],
    test_size=0.2,
    random_state=41)
print(train_features.shape)

correlated_features = set()
correlation_matrix = sperm_data.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)
print(train_features.shape)		

'''
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(train_features)

train_features = constant_filter.transform(train_features)
test_features = constant_filter.transform(test_features)
print(train_features.shape)

qconstant_filter = VarianceThreshold(threshold=0.1)
qconstant_filter.fit(train_features)

train_features = qconstant_filter.transform(train_features)
test_features = qconstant_filter.transform(test_features)
print(train_features.shape)
'''
