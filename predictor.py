import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

def pre_process(sperm_data):
	sperm_data = sperm_data.replace("-", np.nan)
	print(sperm_data.shape)
	sperm_data = sperm_data.dropna()
	print(sperm_data.shape)

	# Create an object to transform the data to fit minmax processor
	scaled_data = StandardScaler().fit_transform(sperm_data.values)
	# Run the normalizer on the dataframe
	norm_data = pd.DataFrame(scaled_data, index=sperm_data.index, columns=sperm_data.columns)
	
	return norm_data
	
def feature_selection(processed_data):
	correlated_features = set()
	correlation_matrix = processed_data.corr()
	for i in range(len(correlation_matrix.columns)):
		for j in range(i):
			if abs(correlation_matrix.iloc[i, j]) > 0.8:
				colname = correlation_matrix.columns[i]
				correlated_features.add(colname)
	processed_data.drop(labels=correlated_features, axis=1, inplace=True)
	print(processed_data.shape)
	
	# Split data to train and test
	train_features, test_features, train_blast, test_blast = train_test_split(
		processed_data.drop(labels=['BLAST_D8'], axis=1),
		processed_data['BLAST_D8'],
		test_size=0.2,
		random_state=41)
	print(train_features.shape)
	
	best_features = SelectKBest(f_regression, k=3).fit_transform(train_features, train_blast)
	print(best_features.shape)
	
	return best_features, train_features, test_features, train_blast, test_blast
	
if __name__ == "__main__":

	raw_data = pd.read_csv(r"all_features.csv")
	processed_data = pre_process(raw_data)
	best_features, train_features, test_features, train_blast, test_blast = feature_selection(processed_data)
	