import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def pre_process(sperm_data):
	sperm_data = sperm_data.replace("-", np.nan)
	print(sperm_data.shape)
	sperm_data = sperm_data.apply(pd.to_numeric, errors='coerce')
	sperm_data = sperm_data.dropna()
	print(sperm_data.shape)
	'''
	# Create an object to transform the data to fit minmax processor
	scaled_data = StandardScaler().fit_transform(sperm_data.values)
	# Run the normalizer on the dataframe
	norm_data = pd.DataFrame(scaled_data, index=sperm_data.index, columns=sperm_data.columns)
	'''
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
		test_size=0.3,
		random_state=41)
	print(train_features.shape)
	'''
	best_features = SelectKBest(f_regression, k=3).fit_transform(train_features, train_blast)
	print(best_features.shape)
	
	return best_features, train_features, test_features, train_blast, test_blast
	'''
	return train_features, test_features, train_blast, test_blast
	
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

	K = RBF(length_scale=0.1)
	gprn = GaussianProcessRegressor(kernel=K, alpha=0.5).fit(train_features, train_blast)
	y = gprn.predict(train_features)
	y_pred = gprn.predict(test_features)

	x_plot = np.linspace(0, 1, len(y))
	X = x_plot.T
	x_plot = np.vstack((x_plot, X))
	for i in range(14):
		x_plot = np.vstack((x_plot, X))
	x_plot = x_plot.T
	print(x_plot.shape)

	plt.gca()
	plt.scatter(np.linspace(0, 1, len(y)), y, c='red', label='prediction')
	plt.scatter(np.linspace(0, 1, len(y)), train_blast, c='blue', label='true')
	plt.plot(x_plot[:,0], gprn.predict(x_plot))
	plt.legend()
	plt.show()
	
	rms = np.sqrt(mean_squared_error(train_blast, y))
	plt.scatter(np.linspace(0, 1, len(y_pred)), y_pred, c='green', label='prediction_t')
	plt.scatter(np.linspace(0, 1, len(y_pred)), test_blast, c='yellow', label='true_t')
	print(rms)
	rms = np.sqrt(mean_squared_error(test_blast, y_pred))
	print(rms)

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
    max_features = 4,
    min_samples_leaf = 1,
    min_samples_split = 7,
    n_estimators = 400)
	rf.fit(train_features, train_blast)
	
	predictions_train = rf.predict(train_features)
	predictions_test = rf.predict(test_features)
	
	calc_error(predictions_train, predictions_test, train_blast, test_blast)
	
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.model_selection import GridSearchCV
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
	rf_random = RandomizedSearchCV(rf, parameters)
	rf_random.fit(train_features, train_blast)
	
	best_grid = rf_random.best_estimator_
	
	predictions_train = best_grid.predict(train_features)
	predictions_test = best_grid.predict(test_features)
	
	#calc_error(predictions_train, predictions_test, train_blast, test_blast)
	
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
	params = rf_random.get_params()
	return rf_random.best_params_, params

def boosted_trees(train_features, train_blast, test_features, test_blast):
	from sklearn.ensemble import GradientBoostingRegressor
	
	br = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1)
	br.fit(train_features, train_blast)
	
	errors = [mean_squared_error(test_blast, y_pred) for y_pred in br.staged_predict(test_features)]
	print(min(errors))
	best_n_est = np.argmin(errors)
	
	print(best_n_est)
	
	br_2 = GradientBoostingRegressor(max_depth=40, n_estimators=25, learning_rate=1)
	br_2.fit(train_features, train_blast)
	
	y_pred = br_2.predict(test_features)
	
	error = mean_absolute_error(test_blast, y_pred)
	print(error)
	calc_error([1,2], y_pred, [1,2], test_blast)
	
	from sklearn.model_selection import RandomizedSearchCV
	from sklearn.model_selection import GridSearchCV
	parameters = {
    'max_features': [3, 4, 5],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [6, 7, 8],
    'n_estimators': [10, 20, 30, 40, 50],
	'max_leaf_nodes': [30,40,50,60,70,80],
	'learning_rate': [0.01, 0.03, 0.05, 0.07]
	}
	
	#br_random = RandomizedSearchCV(br, parameters)
	br_random = GridSearchCV(br, parameters,scoring = make_scorer(mean_absolute_error))
	br_random.fit(train_features, train_blast)
	
	best_grid = br_random.best_estimator_
	
	predictions_test = best_grid.predict(test_features)
	error = mean_absolute_error(test_blast, predictions_test)
	params = br_random.get_params()
	calc_error([1,2], predictions_test, [1,2], test_blast)
	
	print(error)
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
		plt.scatter(np.linspace(0, 1, len(predictions_test)), predictions_test, c='green', label='test prediction')
		plt.scatter(np.linspace(0, 1, len(predictions_test)), test_blast, c='yellow', label='test true')
		plt.legend()
		plt.show()
	
	return None

if __name__ == "__main__":
	raw_data = pd.read_csv(r"all_features.csv")
	processed_data = pre_process(raw_data)
	
	#best_features, train_features, test_features, train_blast, test_blast = feature_selection(processed_data)
	train_features, test_features, train_blast, test_blast = feature_selection(processed_data)
	
	# Saving feature names for later use
	feature_list = list(train_features.columns)
	
	if '--ridge' in sys.argv:
		params = kernel_ridge(train_features, train_blast, test_features, test_blast)
	elif '--forest' in sys.argv:
		params = random_forest(train_features, train_blast, test_features, test_blast)
	elif '--boost' in sys.argv:
		params = boosted_trees(train_features, train_blast, test_features, test_blast)
		
	print(params)