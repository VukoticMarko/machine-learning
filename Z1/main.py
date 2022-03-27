import numpy as np
import pandas

import sys
# import matplotlib.pyplot as plt


# e = 1.718281828459045
class Regression:

    def __init__(self):
        self._t0 = 1
        self._t1 = 1

    def calc_function(self, x):
        # t1 * x^5 + t2
        return np.power(10, (self._t0 * x + self._t1))
        # return self._t0 * x**8 + x**e + self._t1

    def calc_log_function(self, x):
        # t1 * x^5 + t2
        return self._t0 * x + self._t1

    def derivative_t0(self, x, y_predicted, y):
        return (2 / len(x)) * sum((y_predicted - y) * (x))

    def derivative_t1(self, x, y_predicted, y):
        return (2 / len(x)) * sum((y_predicted - y) * (1))

    def mean_squared_error(self, y_predicted, y_true):
        # Calculating the loss or cost
        cost = np.sum((y_predicted - y_true) ** 2) / len(y_true)
        return cost

    def root_mean_squared_error(self, y_predicted, y_true):
        return self.mean_squared_error(y_predicted, y_true) ** 0.5

    # def plot_function(self, from_x, to_x):
    #     x_data = np.linspace(from_x, to_x, 100)
    #     y_data = self.calc_function(x_data)
    #     plt.plot(x_data, y_data)

    def gradient_descent(self, x, y, iterations=20000, learning_rate=0.01,
                         stopping_threshold=1e-8, validation_data=None):

        y = np.log10(y)

        self._t0 = 7.8339
        self._t1 = -1.7869
        # self._t0 = 0
        # self._t1 = 0

        validation_best_t0 = self._t0
        validation_best_t1 = self._t1
        validation_best_cost = 1000

        n = float(len(x))

        costs = []
        weights = []
        previous_cost = None

        for i in range(iterations):

            y_predicted = self.calc_log_function(x)

            current_cost = self.root_mean_squared_error(y_predicted, y)

            if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
                break

            if validation_data is not None:
                valid_y_predicted = self.calc_log_function(validation_data.X)
                validation_current_cost = self.root_mean_squared_error(valid_y_predicted, validation_data.Y)

                if validation_current_cost < validation_best_cost:
                    validation_best_cost = validation_current_cost
                    validation_best_t0 = self._t0
                    validation_best_t1 = self._t1

            previous_cost = current_cost

            costs.append(current_cost)
            weights.append(self._t0)

            # Calculating the gradients
            t0_derivative = self.derivative_t0(x, y_predicted, y)
            t1_derivative = self.derivative_t1(x, y_predicted, y)

            # Updating weights and bias
            self._t0 = self._t0 - (learning_rate * t0_derivative)
            self._t1 = self._t1 - (learning_rate * t1_derivative)

            # Printing the parameters for each 1000th iteration
            print(f"Iteration {i + 1}: Cost: {10**current_cost:<10.5} t0: {self._t0:<10.5} t1: {self._t1:<10.5}")

        # Visualizing the weights and cost at for all iterations
        # plt.plot(weights, costs)
        # plt.title("Gradient descent")
        # plt.ylabel("Cost")
        # plt.xlabel("T0")
        # print("Current cost: " + str(current_cost))

        if validation_data is not None:
            # print("Valid-C-Cost:", validation_current_cost, " | Valid-Best-Cost", validation_best_cost )
            self._t0 = validation_best_t0
            self._t1 = validation_best_t1

        return self._t0, self._t1


def remove_outliers(csv):
    outliers = []
    for i, [x, y] in enumerate(csv.values):
        if 0 < x < 0.4 and y > 30:
            outliers.append(i)

    csv.drop(index=csv.index[outliers], axis=0, inplace=True)


def split_tvt(dataset, validate=0, test=0.5):
    random = np.random.randint(1, 10000)
    # random = 800
    validate = dataset.sample(frac=validate, random_state=random)
    dataset = dataset.drop(validate.index)

    test = dataset.sample(frac=test, random_state=random)
    dataset = dataset.drop(test.index)

    train = dataset
    return [train, validate, test]


def main(argv):
    train_file = "train.csv"
    test_file = "test_preview.csv"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]

    train_data = pandas.read_csv(train_file)
    test_data = pandas.read_csv(test_file)
    remove_outliers(train_data)


    # train_data, va, test_data = split_tvt(train_data)

    # plt.scatter(train_data.X, train_data.Y, edgecolors='red')

    reg = Regression()
    reg.gradient_descent(train_data.X, train_data.Y, validation_data=None)
    # reg.plot_function(0, 0.5)

    # rmse = reg.root_mean_squared_error(reg.calc_function(te.X), te.Y)
    # print(rmse)

    rmse = reg.root_mean_squared_error(reg.calc_function(test_data.X), test_data.Y)
    print(rmse)

    # plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
