import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from math import isnan
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D


def pproc(df):
	# Pre-processing
	df = df.replace("-", np.nan)
	#df[['STR']] = df[['STR']].astype(np.float64)
	#df[['LIN']] = df[['LIN']].astype(np.float64)
	return df
	
def norm(df):
	# Create a minimum and maximum processor object
	min_max_scaler = preprocessing.MinMaxScaler()

	# Create an object to transform the data to fit minmax processor
	x_scaled = min_max_scaler.fit_transform(df)

	# Run the normalizer on the dataframe
	df_normalized = pd.DataFrame(x_scaled)
	return df_normalized

def label(df):
	df = df.replace("-", np.nan)
	df = df.dropna()
	
	brate = pd.to_numeric(df["BLAST_D8"], errors='coerce')

	df.loc[brate >= 25, "class"] = "High"
	df.loc[brate < 25, "class"] = "Low"
	
	class_colours = [r'#3366ff', r'#cc3300', r'#ffc34d']
	colour = []

	labels = list(df['class'])
	for i in labels:
		if i == 'High':
			colour.append(class_colours[0])
		elif i == 'Low':
			colour.append(class_colours[1])
		else:
			colour.append(class_colours[2])
			
	return df, colour
	
def visualise(df_labeled, blast):
	'''
	n_features = df_labeled.shape[1] - 2
	fig, ax = plt.subplots(n_features, n_features)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
	df = df_labeled.values
	
	for j in range(n_features):
		for k in range(n_features):
			ax[k, j].scatter(df[:, k], df[:, j], c=colour,s=0.5)
			ax[k,j].axes.get_xaxis().set_visible(False)
			ax[k,j].axes.get_yaxis().set_visible(False)
	plt.show()
	'''
	df = df_labeled.values
	n_features = df_labeled.shape[1]
	fig, ax = plt.subplots(1, n_features)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
	
	for i in range(n_features):
		ax[i].scatter(df[:,i], blast)
	plt.show()
	
	return None
	
def calc_pca(df, colour):
	blast = df['BLAST_D8']
	label = df['class']
	df = df.drop(['BLAST_D8', 'class'], axis=1)
	
	pca = PCA(n_components=2)
	train_trans = pca.fit_transform(df)

	plt.gca()
	plt.scatter(train_trans[:,0], train_trans[:,1], c=colour)
	plt.show()
	return None
	
def calc_fda(df, colour):
	blast = df['BLAST_D8']
	label = df['class']
	df = df.drop(['BLAST_D8', 'class'], axis=1)
	
	trans = LinearDiscriminantAnalysis()
	trans.fit(df, label)
	
def feature_sel(df, blast):
	X_new = SelectKBest(f_regression, k=2).fit_transform(df, blast)
	
	return X_new
	
def least_squares(X, Y):
    """Calculates the coefficients for equation of regession line."""
    X_T = np.transpose(X)

    a_1 = np.dot(X_T, X)
    a_2 = np.linalg.inv(a_1)
    a_3 = np.dot(a_2, X_T)
    a = np.dot(a_3, Y)

    return a

df = pd.read_csv('motility_features.csv')
print(df.shape)
df = df.apply(pd.to_numeric, errors='coerce')
df = pproc(df)

df = df.dropna()
print(df.shape)

blast = df['BLAST_D8']
print(blast.shape)
df = df.drop('BLAST_D8', axis=1)

visualise(df, blast)

# Covar matrix
C = df.cov()

df_normalized = norm(df)

# Feature Selection
df_new = feature_sel(df_normalized, blast)
print(df_new.shape)

poly = PolynomialFeatures(degree=2)
df_poly = poly.fit_transform(df_new)
weights = least_squares(df_poly, blast)

'''
fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
ax[0, 0].scatter(df_new[:, 0], blast[:])
ax[1, 0].scatter(df_new[:, 1], blast[:])
'''
'''
x1 = np.linspace(0,1,312)
x2 = np.linspace(0,0.2,312)
X_test = np.vstack((x1,x2))
X_test = X_test.T
print(X_test.shape)
poly_test = poly.fit_transform(X_test)
y_test = np.dot(poly_test,weights)
y = np.dot(df_poly, weights)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_new[:,0],df_new[:,1], y)
ax.plot(x1,x2,y_test)

#plt.scatter(df_new[:, 0], df_new[:, 1])
plt.show()
'''

# Normalized covar matrix
C_n = df_normalized.cov()
df_normalized['BLAST_D8'] = blast
df_labeled, colour = label(df_normalized)


#calc_pca(df, colour)
