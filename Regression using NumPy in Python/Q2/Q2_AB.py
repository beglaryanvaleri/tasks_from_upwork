import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 12)


def main():
	print('START Q2_AB\n')

	with open('../datasets/Q1_B_train.txt') as file:  # Read train data
		text = file.readlines()

	x = np.array([])
	y = np.array([])
	for i in range(0, len(text)):
		x = np.append(x, float(text[i].split()[1]))
		y = np.append(y, float(text[i].split()[3]))


	y_mat = np.mat(y)
	print(y_mat)
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

	y_pred = local_weight_regression(data, y_mat, 0.204)
	indic_sort = data[:, 1].argsort(0)
	x_sort = data[indic_sort][:, 0]
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.scatter(x, y, color='green')
	ax.plot(x_sort[:, 1], y_pred[indic_sort], color='red')
	plt.show();



	print('END Q2_AB\n')


if __name__ == "__main__":
    main()
