import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 12)

def main():
	print('START Q3_AB\n')

	with open('../datasets/Q3_data.txt') as file:  # Read data
		text = file.readlines()

	x = np.array([])
	y = np.array([])
	for i in range(0, len(text)):
		x = np.append(x, np.array(float(text[i].split()[1][:-1])))
		x = np.append(x, float(text[i].split()[2][:-1]))
		x = np.append(x, float(text[i].split()[3][:-2]))
		if (text[i].split()[4] == 'M'):
			y = np.append(y, 1.)
		else:
			y = np.append(y, 0.)
	x = x.reshape(len(text), 3)
	y = y.reshape(len(text), 1)

	def compute_p(theta, x, y):
		m = len(y)
		y_pred = logistic_function(np.dot(x, theta))
		err = (((1 - y) * np.log(1 - y_pred) + y * np.log(y_pred)))
		p = -1 / m * sum(err)
		gradient = 1 / m * np.dot(x.transpose(), (y_pred - y))
		return p[0], gradient

	def logistic_function(x):
		return 1 / (1 + np.exp(-x))

	mean_sc = np.mean(x, axis=0)
	std_sc = np.std(x, axis=0)
	scores = (x - mean_sc) / std_sc  # standardization

	rows = scores.shape[0]
	cols = scores.shape[1]

	X = np.append(np.ones((rows, 1)), scores, axis=1)  # include intercept
	y = y.reshape(rows, 1)

	theta_init = np.zeros((cols + 1, 1))
	# cost, gradient = compute_p(theta_init, X, y)

	def gradient_descent(x, y, theta, alpha, iterations):
		p1 = []
		for i in range(iterations):
			p, gradient = compute_p(theta, x, y)
			theta -= (alpha * gradient)
			p1.append(p)
		return theta, p1

	theta, p = gradient_descent(X, y, theta_init, 1, 200)


	def predict(theta, x):
		results = x.dot(theta)
		return results > 0

	p = predict(theta, X)
	print("Training Accuracy:", sum(p == y)[0], "%")
	print("Theta: ", theta)





	print('END Q3_AB\n')
if __name__ == "__main__":
    main()
