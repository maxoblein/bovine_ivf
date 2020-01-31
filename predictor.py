import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

def pre_process(sperm_data):
	sperm_data = sperm_data.replace("-", np.nan)
	print(sperm_data.shape)
	sperm_data = sperm_data.apply(pd.to_numeric, errors='coerce')
	sperm_data = sperm_data.dropna()
	print(sperm_data.shape)
	sperm_data['WOB'] = (sperm_data['VAP'] / sperm_data['VCL']) * 100
	sperm_data['DNC']  = sperm_data['VAP'] * sperm_data['ALH']
	print(sperm_data.shape)
	sperm_data = sperm_data[(sperm_data['VCL'] * sperm_data['ALH']) != 9160.5] # Outlier
	print(sperm_data.shape)

	# Q1 = sperm_data.quantile(0.25)
	# Q3 = sperm_data.quantile(0.75)
	# IQR = Q3 - Q1

	# sperm_data_out = sperm_data[~((sperm_data < (Q1 - 1.5 * IQR)) | (sperm_data > (Q3 + 1.5 * IQR))).all(axis=1)]
	# sperm_data_out.shape


	# # Create an object to transform the data to fit minmax processor
	# scaled_data = StandardScaler().fit_transform(sperm_data.values)
	# # Run the normalizer on the dataframe
	# norm_data = pd.DataFrame(scaled_data, index=sperm_data.index, columns=sperm_data.columns)

	# cols = sperm_data.columns
	# sperm_data_out = pd.DataFrame(normalize(sperm_data, norm='l2'), columns = cols)

	return sperm_data

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
		test_size=0.3)
	print(train_features.shape)


	pca = PCA(n_components=5)
	train_features_pca = pca.fit_transform(train_features)
	test_features_pca = pca.transform(test_features)


	'''
	best_features = SelectKBest(f_regression, k=3).fit_transform(train_features, train_blast)
	print(best_features.shape)

	return best_features, train_features, test_features, train_blast, test_blast
	'''
	return train_features_pca, test_features_pca, train_blast, test_blast

def least_squares(X, Y):
    """Calculates the coefficients for equation of regession line."""
    X_T = np.transpose(X)

    a_1 = np.dot(X_T, X)
    a_2 = np.linalg.inv(a_1)
    a_3 = np.dot(a_2, X_T)
    a = np.dot(a_3, Y)

    return a

def lin_regression(best_features, train_blast):
	poly = PolynomialFeatures(degree=3)
	data_poly = poly.fit_transform(best_features)
	weights = least_squares(data_poly, train_blast)

	y = np.dot(data_poly, weights)
	error = np.linalg.norm(train_blast[:][1] - y[:])
	print(error)

	plt.gca()
	plt.scatter(np.linspace(0, 1, len(y)), y, c='red', label='prediction')
	plt.scatter(np.linspace(0, 1, len(y)), train_blast, c='blue', label='true')
	plt.legend()
	plt.show()

	return weights

def ker_regression(train_features, train_blast):
	rbf_feature = RBFSampler(gamma=1, random_state=1)
	data_rbf = rbf_feature.fit_transform(train_features)
	weights = least_squares(data_rbf, train_blast)

	y = np.dot(data_rbf, weights)
	error = np.linalg.norm(train_blast[:][1] - y[:])
	print(error)

	plt.gca()
	plt.scatter(np.linspace(0, 1, len(y)), y, c='red')
	plt.scatter(np.linspace(0, 1, len(y)), train_blast, c='blue')
	plt.show()

	return weights

def gpr(train_features, train_blast, test_features, test_blast):
	from sklearn.gaussian_process import GaussianProcessRegressor
	from sklearn.gaussian_process.kernels import RBF

	#K = RBF(length_scale=0.1)
	#gprn = GaussianProcessRegressor(kernel=K, alpha=0.5).fit(train_features, train_blast)
	kernel = RBF(10.0, (1e-3, 1e3))
	gprn = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1).fit(train_features, train_blast)
	predictions_train = gprn.predict(train_features)
	predictions_test = gprn.predict(test_features)

	calc_error(predictions_train, predictions_test, train_blast, test_blast)
	#
	# parameters = {
	# 'alpha': [0.2, 0.4, 0.6, 0.8, 1],
	# 'kernel_length_scale': [0.05, 0.1, 0.15],
	# 'kernel': [RBF]}
	#
	# gprn_random = RandomizedSearchCV(gprn, parameters)
	# gprn_random.fit(train_features, train_blast)
	#
	# best_grid = gprn_random.best_estimator_
	#
	# predictions_train = best_grid.predict(train_features)
	# predictions_test = best_grid.predict(test_features)
	#
	# calc_error(predictions_train, predictions_test, train_blast, test_blast)


	params = gprn.get_params()
	return params

def kernel_ridge(train_features, train_blast, test_features, test_blast):
	rbf_feature = KernelRidge(kernel='cosine')
	data_rbf = rbf_feature.fit(train_features, train_blast)

	predictions_train = rbf_feature.predict(train_features)
	predictions_test = rbf_feature.predict(test_features)

	calc_error(predictions_train, predictions_test, train_blast, test_blast)

	params = rbf_feature.get_params()

	return params

def random_forest(train_features, train_blast, test_features, test_blast):
	rf = RandomForestRegressor(random_state = 42,
    bootstrap = False,
    max_depth = 80,
    max_features = 3,
    min_samples_leaf = 1,
    min_samples_split = 6,
	max_leaf_nodes = 60,
    n_estimators = 350)
	rf.fit(train_features, train_blast)

	predictions_train = rf.predict(train_features)
	predictions_test = rf.predict(test_features)

	calc_error(predictions_train, predictions_test, train_blast, test_blast)


	#parameters = {'max_depth':[1,1000], 'max_leaf_nodes':[2,1000], 'n_estimators':[10,2500]}
	'''
	parameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
	}'''
	parameters = {
    'bootstrap': [True, False],
    'max_depth': [75, 80, 85],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [6, 7, 8],
    'n_estimators': [350, 400, 450],
	'max_leaf_nodes': [30,40,50,60,70,80]
	}
	#
	# rf_random = RandomizedSearchCV(rf, parameters)
	# rf_random.fit(train_features, train_blast)
	#
	# best_grid = rf_random.best_estimator_
	#
	# predictions_train = best_grid.predict(train_features)
	# predictions_test = best_grid.predict(test_features)
	#
	# calc_error(predictions_train, predictions_test, train_blast, test_blast)

	'''
	clf = GridSearchCV(rf, parameters)
	clf.fit(train_features, train_blast)

	best_grid = clf.best_estimator_

	predictions_train = best_grid.predict(train_features)
	predictions_test = best_grid.predict(test_features)

	calc_error(predictions_train, predictions_test, train_blast, test_blast)
	'''
	#params = clf.get_params()
	#, clf.best_params_
	params = rf.get_params()
	return params

def boosted_trees(train_features, train_blast, test_features, test_blast):
	from sklearn.ensemble import GradientBoostingRegressor

	bt = GradientBoostingRegressor(max_depth=30, n_estimators=30, learning_rate=0.3, max_features=3, min_samples_leaf=2, min_samples_split=8)
	bt.fit(train_features, train_blast)

	predictions_train = bt.predict(train_features)
	predictions_test = bt.predict(test_features)

	#calc_error(predictions_train, predictions_test, train_blast, test_blast)
	params = bt.get_params()


	parameters = {
    'max_features': [3, 4, 5],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [6, 7, 8],
    'n_estimators': [10, 20, 30, 40, 50,60,70,80,90],
	'max_depth': [30,40,50,60,70,80],
	'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.3,0.4,0.5]
	}

	bt_random = RandomizedSearchCV(bt, parameters, scoring = make_scorer(bal_error, greater_is_better=False))
	bt_random.fit(train_features, train_blast)

	best_grid = bt_random.best_estimator_

	predictions_train = best_grid.predict(train_features)
	predictions_test = best_grid.predict(test_features)

	calc_error(predictions_train, predictions_test, train_blast, test_blast)
	params = best_grid.get_params()

	return params

def calc_error(predictions_train, predictions_test, train_blast, test_blast):
	'''
	errors = abs(predictions_train - train_blast)
	print('Mean Absolute Error:', round(np.mean(errors), 2))

	errors = abs(predictions_test - test_blast)
	print('Mean Absolute Error:', round(np.mean(errors), 2))

	# Calculate mean absolute percentage error (MAPE)
	mape = 100 * (errors / test_blast)

	# Calculate and display accuracy
	accuracy = 100 - mape
	print('Accuracy: ', round(accuracy, 2), '%.')
	'''

	errors = np.abs(test_blast - predictions_test)

	max_err = np.max(np.abs(test_blast - predictions_test))
	print('Max: ', max_err)
	var_err = explained_variance_score(test_blast, predictions_test)
	print('Var: ', var_err)
	mape = mean_absolute_error(test_blast, predictions_test)
	print('Mape: ', mape)

	if '--plot' in sys.argv:
		plt.gca()
		#plt.scatter(np.linspace(0, 1, len(predictions_train)), predictions_train, c='red', label='train prediction')
		#plt.scatter(np.linspace(0, 1, len(predictions_train)), train_blast, c='blue', label='train true')
		# plt.scatter(np.linspace(0, 1, len(predictions_test)), predictions_test, c='red', label='test prediction')
		# plt.scatter(np.linspace(0, 1, len(predictions_test)), test_blast, c='blue', label='test true',alpha = 0.5)
		plt.scatter(np.linspace(0, 1, len(predictions_test)),errors,label = 'errors')
		plt.legend()
		plt.show()

	return None

def max_error(test_blast, predictions_test):
	err = np.max(np.abs(test_blast - predictions_test))
	return err

def bal_error(test_blast, predictions_test):
	max = np.max(np.abs(test_blast - predictions_test))
	mae = mean_absolute_error(test_blast, predictions_test)

	a = mae / (mae + max); b = max / (mae + max)
	bal = np.mean([2*a, b])

	return bal

def visualise(df, blast):
	n_features = df.shape[1]
	fig, ax = plt.subplots(1, n_features)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
	df = df.values

	for k in range(n_features):
		ax[k].scatter(df[:, k], blast, s=0.5)
		ax[k].axes.get_xaxis().set_visible(True)
		ax[k].axes.get_yaxis().set_visible(True)
	plt.show()

	return None

if __name__ == "__main__":
	raw_data = pd.read_csv(r"all_features.csv")
	processed_data = pre_process(raw_data)

	#best_features, train_features, test_features, train_blast, test_blast = feature_selection(processed_data)
	train_features, test_features, train_blast, test_blast = feature_selection(processed_data)

	# Saving feature names for later use
	#feature_list = list(train_features.columns)

	if '--ridge' in sys.argv:
		params = kernel_ridge(train_features, train_blast, test_features, test_blast)
	elif '--forest' in sys.argv:
		params = random_forest(train_features, train_blast, test_features, test_blast)
	elif '--boost' in sys.argv:
		params = boosted_trees(train_features, train_blast, test_features, test_blast)
	elif '--gp' in sys.argv:
		params = gpr(train_features, train_blast, test_features, test_blast)
	#visualise(processed_data.drop(['BLAST_D8'], axis=1), processed_data['BLAST_D8'])
	print(params)
