import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from math import isnan
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
	
def visualise(df_labeled):
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
	plt.gca
	plt.scatter(df[:, 2], df[:, 4], c=colour)
	plt.show()
	'''
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

df = pd.read_csv('all_features.csv')
df = pproc(df)
blast = df['BLAST_D8']
df = df.drop('BLAST_D8', axis=1)


# Covar matrix
C = df.cov()

df_normalized = norm(df)

# Normalized covar matrix
C_n = df_normalized.cov()
df_normalized['BLAST_D8'] = blast
df_labeled, colour = label(df_normalized)

#visualise(df_labeled)
calc_pca(df, colour)