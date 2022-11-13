import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 12)

def main():
	print('START Q2_D\n')

	with open('../datasets/Q1_B_train.txt') as file:  # Read train data
		text = file.readlines()

	x = np.array([])
	y = np.array([])
	for i in range(0, 20):
		x = np.append(x, float(text[i].split()[1]))
		y = np.append(y, float(text[i].split()[3]))

	with open('../datasets/Q1_C_test.txt') as file:  # Read test data
		text = file.readlines()

	x_test = np.array([])
	y_test = np.array([])
	for i in range(0, len(text)):
		x_test = np.append(x_test, float(text[i].split()[1]))
		y_test = np.append(y_test, float(text[i].split()[3]))

	data_test = np.hstack((np.ones((len(x_test), 1)), np.mat(x_test).T))

	y_mat = np.mat(y)
	data = np.hstack((np.ones((len(x), 1)), np.mat(x).T))

	def kernel(data, point, x_mat, k):
		m, n = np.shape(x_mat)
		ws = np.mat(np.eye((m)))
		for j in range(m):
			diff = point - data[j]
			ws[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
		return ws

	def local_weight(data, point, x_mat, ymat, k):
		wei = kernel(data, point, x_mat, k)
		return (data.T * (wei * data)).I * (data.T * (wei * ymat.T))

	def local_weight_regression(data, y_mat, k):
		m, n = np.shape(data)
		y_pred = np.zeros(m)
		for i in range(m):
			y_pred[i] = data[i] * local_weight(data, data[i], data, y_mat, k)
		return y_pred

	def local_weight_regression_predict(data_test, data, y_mat, k):
		m_pred, n_pred = np.shape(data_test)
		y_pred = np.zeros(m_pred)
		for i in range(m_pred):
			y_pred[i] = data_test[i] * local_weight(data, data_test[i], data, y_mat, k)
		return y_pred

	y_pred_test = local_weight_regression_predict(data_test, data, y_mat, 0.204)

	error = 0
	for i, j in zip(y_pred_test, y_test):
		error += (i -j) * (i -j)
	print('error:', error)

	y_pred = local_weight_regression(data, y_mat, 0.204)
	indic_sort = data[:, 1].argsort(0)
	x_sort = data[indic_sort][:, 0]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x, y, color='green')
	ax.scatter(x_test, y_pred_test, color='blue')
	ax.plot(x_sort[:, 1], y_pred[indic_sort], color='red')
	plt.show();

	print('END Q2_D\n')


if __name__ == "__main__":
    main()

# Comparing the results with the 1 question D, we can say that it is a worse.
# Perhaps this is due to the fact that in this case we need to have at least a few
# points in the neighborhood in order to have a good predict