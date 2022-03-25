import numpy as np
import pandas

import sys
# import matplotlib.pyplot as plt


class Regression:

    def __init__(self):
        self._t0 = 1
        self._t1 = 1

    def calc_function(self, x):
        # t1 * x^5 + t2
        return self._t0 * x**6 + self._t1

    def derivative_t0(self, x, y, y_predicted):
        return -(2 / len(x)) * sum(6 * x * (y - y_predicted))

    def derivative_t1(self, y, y_predicted):
        return -(2 / len(y)) * sum(y - y_predicted)

    def mean_squared_error(self, y_true, y_predicted):
        # Calculating the loss or cost
        cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
        return cost

    def root_mean_squared_error(self, y_true, y_predicted):
        return self.mean_squared_error(y_true, y_predicted) ** 0.5

    # def plot_function(self, from_x, to_x):
    #     x_data = np.linspace(from_x, to_x, 100)
    #     y_data = reg.calc_function(x_data)
    #     plt.plot(x_data, y_data)

    def gradient_descent(self, x, y, iterations=12000, learning_rate=0.2,
                         stopping_threshold=1e-6, validation_data=None):
        self._t0 = 0.1
        self._t1 = 0.01

        validation_best_t0 = self._t0
        validation_best_t1 = self._t1
        validation_best_cost = 1000

        n = float(len(x))

        costs = []
        weights = []
        previous_cost = None

        for i in range(iterations):

            y_predicted = self.calc_function(x)

            current_cost = self.root_mean_squared_error(y, y_predicted)

            if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
                break

            if validation_data is not None:
                valid_y_predicted = self.calc_function(validation_data.X)
                validation_current_cost = self.root_mean_squared_error(validation_data.Y, valid_y_predicted)

                if validation_current_cost < validation_best_cost:
                    validation_best_cost = validation_current_cost
                    validation_best_t0 = self._t0
                    validation_best_t1 = self._t1

            previous_cost = current_cost

            costs.append(current_cost)
            weights.append(self._t0)

            # Calculating the gradients
            t0_derivative = self.derivative_t0(x, y, y_predicted)
            t1_derivative = self.derivative_t1(y, y_predicted)

            # Updating weights and bias
            self._t0 = self._t0 - (learning_rate * t0_derivative)
            self._t1 = self._t1 - (learning_rate * t1_derivative)

            # Printing the parameters for each 1000th iteration
            # print(f"Iteration {i + 1}: Cost: {current_cost:<10.5} t0: {self._t0:<10.5} t1: {self._t1:<10.5}")

        # Visualizing the weights and cost at for all iterations
        # plt.figure(figsize=(8, 6))
        # plt.plot(weights, costs)
        # plt.scatter(weights, costs, marker='o', color='red')
        # plt.title("Cost vs Weights")
        # plt.ylabel("Cost")
        # plt.xlabel("Weight")
        # plt.show()
        # print("Current cost: " + str(current_cost))

        if validation_data is not None:
            print("Valid-C-Cost:", validation_current_cost, " | Valid-Best-Cost", validation_best_cost )
            self._t0 = validation_best_t0
            self._t1 = validation_best_t1

        return self._t0, self._t1


def remove_outliers(csv):
    outliers = []
    for i, [x, y] in enumerate(csv.values):
        if 0 < x < 1 and y > 30:
            outliers.append(i)

    csv.drop(index=csv.index[outliers], axis=0, inplace=True)


def split_tvt(dataset, validate=0.15, test=0):
    # random = np.random.randint(1, 10000)
    random = 800
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

    # train_data, va, te = split_tvt(train_data)

    # plt.scatter(train_data.X, train_data.Y)

    reg = Regression()
    reg.gradient_descent(train_data.X, train_data.Y, validation_data=None)
    # reg.plot_function(0, 0.4)

    # rmse = reg.root_mean_squared_error(te.Y, reg.calc_function(te.X))
    # print(rmse)

    rmse = reg.root_mean_squared_error(test_data.Y, reg.calc_function(test_data.X))
    print(rmse)

    # plt.show()



if __name__ == '__main__':
    main(sys.argv[1:])
