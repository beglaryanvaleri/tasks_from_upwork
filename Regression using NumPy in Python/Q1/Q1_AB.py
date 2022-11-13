import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 12)

def main():
	print('START Q1_AB\n')

	def  linear_regression_learner(k, d, x, y):  # Function for linear regression learner
		t = [0] * (d + 1) # parameter vector Θ
		L = 0.001  # The learning Rate
		epochs = 2000  # The number of iterations to perform gradient descent

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

		error = [] # list for errors of each epochs
		for epochs in range(epochs): # epoch iteration
			D_t = [0] * (d + 1)  # error storage variable
			for tt in range(0, len(t)): # parameter vector Θ iteration
				for i in range(0, len(x)): # object iteration
					D_t[tt] += y_error(tt, k, x[i], y[i], t)
				t[tt] -= L * D_t[tt] / len(x)  # override parameter
			error1 = 0
			for i in range(0, len(x)):
				error1 += (y_pred(d, k, x[i], t) - y[i]) * (y_pred(d, k, x[i], t) - y[i])
			error.append(error1)

		y_predict = [] # predict
		for i in x:
			y_predict.append(y_pred(d, k, i, t))

		return y_predict, error # Return predicted values and array of error

	with open('../datasets/Q1_B_train.txt') as file: # Read file
		text = file.readlines()

	x = []
	y = []
	for i in range(0, 20):  # for i in range(0, len(text)):
		x.append(float(text[i].split()[1]))
		y.append(float(text[i].split()[3]))

	# Predict values for depth in (0, 6)
	y_predict_0, _ = linear_regression_learner(2, 0, x, y)
	y_predict_1, _ = linear_regression_learner(2, 1, x, y)
	y_predict_2, _ = linear_regression_learner(2, 2, x, y)
	y_predict_3, _ = linear_regression_learner(2, 3, x, y)
	y_predict_4, _ = linear_regression_learner(2, 4, x, y)
	y_predict_5, _ = linear_regression_learner(2, 5, x, y)
	y_predict_6, _ = linear_regression_learner(2, 6, x, y)

	# Show results, you can comment some lines to see more detal in graph
	plt.scatter(x, y, label='Real y')
	plt.scatter(x, y_predict_0, label='Predict d=0')
	plt.scatter(x, y_predict_1, label='Predict d=1')
	plt.scatter(x, y_predict_2, label='Predict d=2')
	plt.scatter(x, y_predict_3, label='Predict d=3')
	plt.scatter(x, y_predict_4, label='Predict d=4')
	plt.scatter(x, y_predict_5, label='Predict d=5')
	plt.scatter(x, y_predict_6, label='Predict d=6')
	plt.legend()
	plt.show()

	print('END Q1_AB\n')


if __name__ == "__main__":
    main()











#
# for i in range(epochs):
#     Y_pred = m * X + m1 * X * X + c  # The current predicted value of Y
#     D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_m1 = (-4 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m1
#     D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     m1 = m1 - L * D_m1  # Update m
#     c = c - L * D_c  # Update c
#
# print(t)