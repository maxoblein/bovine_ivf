import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn_extensions.kernel_regression import KernelRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import GradientBoostingRegressor

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
	#
	# sperm_data_out = sperm_data[~((sperm_data < (Q1 - 1.5 * IQR)) | (sperm_data > (Q3 + 1.5 * IQR))).any(axis=1)]
	# sperm_data_out.shape


	# # Create an object to transform the data to fit minmax processor
	# scaled_data = StandardScaler().fit_transform(sperm_data.values)
	# # Run the normalizer on the dataframe
	# norm_data = pd.DataFrame(scaled_data, index=sperm_data.index, columns=sperm_data.columns)


	# blast = sperm_data['BLAST_D8']
	# sperm_data.drop(['BLAST_D8'], axis=1)
	cols = sperm_data.columns
	# scaler = MinMaxScaler()
	#
	# #sperm_data = pd.DataFrame(scaler.fit_transform(sperm_data), columns = cols)
	# #
	# sperm_data = pd.DataFrame(normalize(sperm_data, norm='l2'), columns = cols)
	# sperm_data['BLAST_D8'] = blast

	return sperm_data, cols

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
		test_size=0.3, random_state=42)
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

def lin_regression(train_features, train_blast, test_features, test_blast):
	regr = linear_model.LinearRegression()
	regr.fit(train_features, train_blast)
	y = regr.predict(test_features)

	calc_error(test_blast, y)
	params = regr.get_params()

	return params

def kernel_reg(train_features, train_blast, test_features, test_blast):
	kernel = KernelRegression(kernel="chi2")
	y = kernel.fit(train_features, train_blast).predict(test_features)

	calc_error(y, test_blast)

	params = kernel.get_params()

	return params

def kernel_ridge(train_features, train_blast, test_features, test_blast):
	kernel = KernelRidge(kernel='additive_chi2')
	y = kernel.fit(train_features, train_blast).predict(test_features)

	calc_error(y, test_blast)
	params = kernel.get_params()

	return params

def gpr(train_features, train_blast, test_features, test_blast):
	#K = RBF(length_scale=0.1)
	#gprn = GaussianProcessRegressor(kernel=K, alpha=0.5).fit(train_features, train_blast)
	kernel = RBF(10.0, (1e-3, 1e3))
	gprn = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1).fit(train_features, train_blast)

	predictions_test = gprn.predict(test_features)

	calc_error(predictions_test, test_blast)
	params = gprn.get_params()

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

	predictions_test = rf.predict(test_features)

	calc_error(predictions_test, test_blast)

	parameters = {
    'bootstrap': [True, False],
    'max_depth': [70, 75, 80, 85, 90, 95, 100, 105, 110],
    'max_features': [1, 2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [6, 7, 8],
    'n_estimators': [350, 400, 450],
	'max_leaf_nodes': [30,40,50,60,70,80]
	}

	rf_random = RandomizedSearchCV(rf, parameters, scoring = make_scorer(bal_error, greater_is_better=False))
	rf_random.fit(train_features, train_blast)

	best_grid = rf_random.best_estimator_

	predictions_test = best_grid.predict(test_features)

	calc_error(predictions_test, test_blast)

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
	params = best_grid.get_params()
	return params

def boosted_trees(train_features, train_blast, test_features, test_blast):
	bt = GradientBoostingRegressor(learning_rate=0.3, max_features=4, random_state=42, subsample=0.8, loss='ls')
	modelfit(bt, train_features, train_blast, performCV=True, printFeatureImportance=True, cv_folds=5)

	predictions_test = bt.predict(test_features)

	calc_error(predictions_test, test_blast)
	#params = bt.get_params()

	parameters = {
    'max_depth': [80, 90, 100, 105, 110, 120, 130, 140],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [90, 100, 150, 200, 250, 300, 350]
	}

	bt_random = RandomizedSearchCV(bt, parameters, scoring=make_scorer(mean_squared_error, greater_is_better=False))
	modelfit(bt_random, train_features, train_blast, performCV=True, printFeatureImportance=True, cv_folds=5)

	bt_random.fit(train_features, train_blast)

	best_grid = bt_random.best_estimator_

	predictions_test = best_grid.predict(test_features)

	calc_error(predictions_test, test_blast)
	params = best_grid.get_params()

	pred_blast = GradientBoostingRegressor(**params).fit(train_features, train_blast).predict(test_features)
	calc_error(pred_blast, test_blast)

	return params

def modelfit(alg, dtrain, train_blast, performCV=True, printFeatureImportance=True, cv_folds=5):
	#Fit the algorithm on the data
	alg.fit(dtrain, train_blast)

	#Predict training set:
	dtrain_predictions = alg.predict(dtrain)

	#Perform cross-validation:
	if performCV:
		cv_score = cross_val_score(alg, dtrain, train_blast, cv=cv_folds)

	#Print model report:
	print("Model Report")
	print("Mean Squared Error (Train): %.4g" % mean_squared_error(train_blast.values, dtrain_predictions))
	print("Max Error (Train): %f" % max_error(train_blast, dtrain_predictions))

	if performCV:
		print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
	'''
	#Print Feature Importance:
	if printFeatureImportance:
		feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
		feat_imp.plot(kind='bar', title='Feature Importances')
		plt.ylabel('Feature Importance Score')
	'''
	return None

def calc_error(predictions_test, test_blast):
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
	# var_err = explained_variance_score(test_blast, predictions_test)
	# print('Var: ', var_err)
	mape = mean_absolute_error(test_blast, predictions_test)
	print('Mae: ', mape)
	mse = mean_squared_error(test_blast, predictions_test)
	print('MSE: ', mse)

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
	bal = np.mean([a, b])

	return bal

def visualise(df, blast):
	n_features = df.shape[1]
	fig, ax = plt.subplots(n_features, n_features)
	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
	df = df.values

	for i in range(n_features):
			for j in range(n_features):
				ax[i, j].scatter(df[:, i], df[:, j], s=0.5)
				ax[i, j].axes.get_xaxis().set_visible(False)
				ax[i, j].axes.get_yaxis().set_visible(False)
	plt.show()

	return None

if __name__ == "__main__":
	raw_data = pd.read_csv(r"all_features.csv")
	processed_data, cols = pre_process(raw_data)

	#best_features, train_features, test_features, train_blast, test_blast = feature_selection(processed_data)
	train_features, test_features, train_blast, test_blast = feature_selection(processed_data)

	# Saving feature names for later use
	#feature_list = list(train_features.columns)
	if '--lin' in sys.argv:
		params = lin_regression(train_features, train_blast, test_features, test_blast)
	elif '--kernel' in sys.argv:
		params = kernel_reg(train_features, train_blast, test_features, test_blast)
	elif '--ridge' in sys.argv:
		params = kernel_ridge(train_features, train_blast, test_features, test_blast)
	elif '--gp' in sys.argv:
		params = gpr(train_features, train_blast, test_features, test_blast)
	elif '--forest' in sys.argv:
		params = random_forest(train_features, train_blast, test_features, test_blast)
	elif '--boost' in sys.argv:
		params = boosted_trees(train_features, train_blast, test_features, test_blast)
		#modelfit(alg, train_features, train_blast, cols, performCV=True, printFeatureImportance=True, cv_folds=5)
	if '--vis' in sys.argv:
		visualise(processed_data.drop(['BLAST_D8'], axis=1), processed_data['BLAST_D8'])

	print(params)
