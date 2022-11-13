import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 12)

def main():
	print('START Q1_D\n')

	with open('../datasets/Q1_C_test.txt') as file: # Read test data
		text = file.readlines()

	x_test = []
	y_test = []
	for i in range(0, len(text)):
		x_test.append(float(text[i].split()[1]))
		y_test.append(float(text[i].split()[3]))


	def  linear_regression_learner(k, d, x_test, y_test):

		with open('../datasets/Q1_B_train.txt') as file: # Read train data
			text = file.readlines()

		x = []
		y = []
		for i in range(0, 20):
			x.append(float(text[i].split()[1]))
			y.append(float(text[i].split()[3]))


		t = [0] * (d + 1) # Parameter vector Θ
		L = 0.001  # The learning Rate
		epochs = 2000  # The number of iterations to perform gradient descent
		n = float(len(x))  # Number of elements in X

		def y_pred(d, k, x, t): # function for predict y
			y_pred = t[0]
			for i in range(1, d + 1):
				y_pred += t[i] * np.sin(i * k * x) * np.sin(i * k * x)
			return y_pred

		def y_error(d, k, x, y, t): # error solving function for one element
			if d == 0:
				y_error = -2 * (y - y_pred(d, k, x, t))
			else:
				y_error = -2 * np.sin(k * d * x) * (y - y_pred(len(t) - 1, k, x, t))
			return y_error

		error = [] # list for errors on train data
		for epochs in range(epochs):
			D_t = [0] * (d + 1)
			for tt in range(0, len(t)):
				for i in range(0, len(x)):
					D_t[tt] += y_error(tt, k, x[i], y[i], t)
				t[tt] -= L * D_t[tt] / len(x)
			error1 = 0
			for i in range(0, len(x)):
				error1 += (y_pred(d, k, x[i], t) - y[i]) * (y_pred(d, k, x[i], t) - y[i])
			error.append(error1)

		y_predict = [] # predict test data
		error_test = 0
		for i in range(0, len(x_test)):
			y_predict.append(y_pred(d, k, x_test[i], t))
			error_test += (y_test[i] - y_pred(d, k, x_test[i], t)) * (y_test[i] - y_pred(d, k, x_test[i], t))

		return y_predict, error[-1], error_test # Return predicted values, error on train data and error on test data



	k = 1 # set k
	d = 6 # set depth
	list_d = [0] * d # list for values of d
	list_k = [0] * d # list for values of k
	list_error_train = [0] * d # list for values of d
	list_error_test = [0] * d # list for values of d
	i = 0
	y_predict = []
	for dd in range(1, d+1): # Determine the results
		kk = k
		list_d[i] = dd
		list_k[i] = kk
		predict, list_error_train[i], list_error_test[i] = linear_regression_learner(kk, dd, x_test, y_test)
		y_predict.append(predict)
		i += 1

	#print results
	for d, k, error_train, error_test in zip(list_d, list_k, list_error_train, list_error_test):
		print('d =', d, 'k=', k, 'error_train=', error_train, 'error_test=', error_test)

	# Plot predicted values
	plt.scatter(x_test, y_test, label='Real y')
	# plt.scatter(x_test, y_predict[0], label='Predict d=1')
	# plt.scatter(x_test, y_predict[1], label='Predict d=2')
	plt.scatter(x_test, y_predict[2], label='Predict d=3')
	plt.scatter(x_test, y_predict[3], label='Predict d=4')
	# plt.scatter(x_test, y_predict[4], label='Predict d=5')
	# plt.scatter(x_test, y_predict[5], label='Predict d=6')
	plt.legend()
	plt.show()


	print('END Q1_D\n')


if __name__ == "__main__":
    main()

# We see little difference between the results, probably because
# these 20 points of view describe well the behavior of all the data.