import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isnan
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

def least_squares(X, Y):
    """Calculates the coefficients for equation of regession line."""
    X_T = np.transpose(X)

    a_1 = np.dot(X_T, X)
    a_2 = np.linalg.inv(a_1)
    a_3 = np.dot(a_2, X_T)
    a = np.dot(a_3, Y)

    return a
	
def norm(df):
	# Create a minimum and maximum processor object
	min_max_scaler = preprocessing.MinMaxScaler()

	# Create an object to transform the data to fit minmax processor
	x_scaled = min_max_scaler.fit_transform(df)

	# Run the normalizer on the dataframe
	df_normalized = pd.DataFrame(x_scaled)
	return df_normalized

sperm_data = pd.read_csv('motility_2.csv')
print(sperm_data.shape)
sperm_data = sperm_data.apply(pd.to_numeric, errors='coerce')
sperm_data = sperm_data.replace("-", np.nan)
sperm_data = sperm_data.dropna()
print(sperm_data.shape)

# Split data to train and test
train_features, test_features, train_blast, test_blast = train_test_split(
    sperm_data.drop(labels=['BLAST_D8'], axis=1),
    sperm_data['BLAST_D8'],
    test_size=0.2,
    random_state=41)
print(train_features.shape)

train_features = norm(train_features)
test_features = norm(test_features)

poly = PolynomialFeatures(degree=3)
data_poly = poly.fit_transform(train_features)
weights = least_squares(data_poly, train_blast)
print(weights)

'''
x1 = np.linspace(0,1,249)
x2 = np.linspace(0,0.2,249)
X_test = np.vstack((x1,x2))
X_test = X_test.T
print(X_test.shape)

poly_test = poly.fit_transform(X_test)
y_test = np.dot(poly_test,weights)
'''

y = np.dot(data_poly, weights)

'''
test_poly = poly.fit_transform(test_features)
pred_blast = 
'''

e = np.linalg.norm(train_blast[:][1] - y[:])
print(e)
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_features[:][0], train_features[:][1], train_blast)
ax.scatter(train_features[:][0], train_features[:][1], y, c='red')
#ax.plot(x1,x2,y_test)

#plt.scatter(df_new[:, 0], df_new[:, 1])
'''
print(len(np.linspace(0, 1, len(y))))
print(train_blast.shape)
plt.gca()
plt.scatter(np.linspace(0, 1, len(y)), y, c='red')
plt.scatter(np.linspace(0, 1, len(y)), train_blast, c='blue')
plt.show()