import numpy as np
import pandas

import sys
# import matplotlib.pyplot as plt


class Regression:

    def __init__(self):
        self._t_musko_0 = 1
        self._t_zensko_0 = 1

        self._t_god_iskustva_0 = 1
        self._t_god_iskustva_1 = 1
        self._t_god_iskustva_2 = 1

        self._t_godina_doktor_0 = 1
        self._t_godina_doktor_1 = 1
        self._t_godina_doktor_2 = 1

        self._t_asst_prof_0 = 1
        self._t_assoc_prof_0 = 1
        self._t_oblast_a_0 = 1

    def calc_function(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof, x_oblast_a):
        return self._t_musko_0 * x_musko + self._t_zensko_0 * x_zensko + self._t_god_iskustva_0 + self._t_god_iskustva_1 * x_god_iskustva + self._t_god_iskustva_2 * x_god_iskustva ** 2 + self._t_godina_doktor_0 + self._t_godina_doktor_1 * x_godina_doktor + self._t_godina_doktor_1 * x_godina_doktor ** 2 + self._t_asst_prof_0 * x_asst_prof + self._t_assoc_prof_0 * x_assoc_prof + self._t_oblast_a_0 * x_oblast_a

    def derivative_t_musko_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                             x_oblast_a, y_predicted, y):
        return (2 / len(x_musko)) * sum((y_predicted - y) * (x_musko))

    def derivative_t_zensko_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                              x_oblast_a, y_predicted, y):
        return (2 / len(x_zensko)) * sum((y_predicted - y) * (x_zensko))

    def derivative_t_god_iskustva_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                    x_oblast_a, y_predicted, y):
        return (2 / len(x_god_iskustva)) * sum((y_predicted - y) * (1))

    def derivative_t_god_iskustva_1(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                    x_oblast_a, y_predicted, y):
        return (2 / len(x_god_iskustva)) * sum((y_predicted - y) * (x_god_iskustva))

    def derivative_t_god_iskustva_2(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                    x_oblast_a, y_predicted, y):
        return (2 / len(x_god_iskustva)) * sum((y_predicted - y) * (2 * x_god_iskustva))

    def derivative_t_godina_doktor_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof,
                                     x_assoc_prof, x_oblast_a, y_predicted, y):
        return (2 / len(x_godina_doktor)) * sum((y_predicted - y) * (1))

    def derivative_t_godina_doktor_1(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof,
                                     x_assoc_prof, x_oblast_a, y_predicted, y):
        return (2 / len(x_godina_doktor)) * sum((y_predicted - y) * (x_godina_doktor))

    def derivative_t_godina_doktor_2(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof,
                                     x_assoc_prof, x_oblast_a, y_predicted, y):
        return (2 / len(x_godina_doktor)) * sum((y_predicted - y) * (2 * x_godina_doktor))

    def derivative_t_asst_prof_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                 x_oblast_a, y_predicted, y):
        return (2 / len(x_asst_prof)) * sum((y_predicted - y) * (x_asst_prof))

    def derivative_t_assoc_prof_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                  x_oblast_a, y_predicted, y):
        return (2 / len(x_assoc_prof)) * sum((y_predicted - y) * (x_assoc_prof))

    def derivative_t_oblast_a_0(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                                x_oblast_a, y_predicted, y):
        return (2 / len(x_oblast_a)) * sum((y_predicted - y) * (x_oblast_a))

    def mean_squared_error(self, y_predicted, y_true):
        # Calculating the loss or cost
        cost = np.sum((y_predicted - y_true) ** 2) / len(y_true)
        return cost

    def root_mean_squared_error(self, y_predicted, y_true):
        return self.mean_squared_error(y_predicted, y_true) ** 0.5

    def gradient_descent(self, x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof, x_assoc_prof,
                         x_oblast_a, y, iterations=1000, learning_rate=0.00045,
                         stopping_threshold=1e-8, validation_data=None):
        # plt.scatter(x, y, edgecolors='red')

        self._t_musko_0 = 44381.4203488993
        self._t_zensko_0 = 36078.85584814751
        self._t_god_iskustva_0 = 41386.77988696933
        self._t_god_iskustva_1 = -15.358051978892055
        self._t_god_iskustva_2 = -15.724061814251009
        self._t_godina_doktor_0 = 47196.72825912894
        self._t_godina_doktor_1 = 11.599840323871948
        self._t_godina_doktor_2 = 11.24973362867113
        self._t_asst_prof_0 = -46687.80313476979
        self._t_assoc_prof_0 = -32593.776756540374
        self._t_oblast_a_0 = -12144.945768550824


        n = float(len(x_musko))

        costs = []
        weights = []

        for i in range(iterations):
            y_predicted = self.calc_function(x_musko, x_zensko, x_god_iskustva, x_godina_doktor, x_asst_prof,
                                             x_assoc_prof, x_oblast_a)

            current_cost = self.root_mean_squared_error(y_predicted, y)

            costs.append(current_cost)
            weights.append(self._t_god_iskustva_0)

            # Calculating the gradients
            derivative_t_musko_0 = self.derivative_t_musko_0(x_musko, x_zensko, x_god_iskustva, x_godina_doktor,
                                                             x_asst_prof, x_assoc_prof, x_oblast_a, y_predicted, y)
            derivative_t_zensko_0 = self.derivative_t_zensko_0(x_musko, x_zensko, x_god_iskustva, x_godina_doktor,
                                                               x_asst_prof, x_assoc_prof, x_oblast_a, y_predicted, y)

            derivative_t_god_iskustva_0 = self.derivative_t_god_iskustva_0(x_musko, x_zensko, x_god_iskustva,
                                                                           x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                           x_oblast_a, y_predicted, y)
            derivative_t_god_iskustva_1 = self.derivative_t_god_iskustva_1(x_musko, x_zensko, x_god_iskustva,
                                                                           x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                           x_oblast_a, y_predicted, y)
            derivative_t_god_iskustva_2 = self.derivative_t_god_iskustva_2(x_musko, x_zensko, x_god_iskustva,
                                                                           x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                           x_oblast_a, y_predicted, y)

            derivative_t_godina_doktor_0 = self.derivative_t_godina_doktor_0(x_musko, x_zensko, x_god_iskustva,
                                                                             x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                             x_oblast_a, y_predicted, y)
            derivative_t_godina_doktor_1 = self.derivative_t_godina_doktor_1(x_musko, x_zensko, x_god_iskustva,
                                                                             x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                             x_oblast_a, y_predicted, y)
            derivative_t_godina_doktor_2 = self.derivative_t_godina_doktor_2(x_musko, x_zensko, x_god_iskustva,
                                                                             x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                             x_oblast_a, y_predicted, y)

            derivative_t_asst_prof_0 = self.derivative_t_asst_prof_0(x_musko, x_zensko, x_god_iskustva, x_godina_doktor,
                                                                     x_asst_prof, x_assoc_prof, x_oblast_a, y_predicted,
                                                                     y)
            derivative_t_assoc_prof_0 = self.derivative_t_assoc_prof_0(x_musko, x_zensko, x_god_iskustva,
                                                                       x_godina_doktor, x_asst_prof, x_assoc_prof,
                                                                       x_oblast_a, y_predicted, y)
            derivative_t_oblast_a_0 = self.derivative_t_oblast_a_0(x_musko, x_zensko, x_god_iskustva, x_godina_doktor,
                                                                   x_asst_prof, x_assoc_prof, x_oblast_a, y_predicted,
                                                                   y)

            # Updating weights and bias
            self._t_musko_0 = self._t_musko_0 - (learning_rate * derivative_t_musko_0)
            self._t_zensko_0 = self._t_zensko_0 - (learning_rate * derivative_t_zensko_0)
            self._t_god_iskustva_0 = self._t_god_iskustva_0 - (learning_rate * derivative_t_god_iskustva_0)
            self._t_god_iskustva_1 = self._t_god_iskustva_1 - (learning_rate * 0.01 * derivative_t_god_iskustva_1)
            self._t_god_iskustva_2 = self._t_god_iskustva_2 - (learning_rate * 0.001 * derivative_t_god_iskustva_2)

            self._t_godina_doktor_0 = self._t_godina_doktor_0 - (learning_rate * derivative_t_godina_doktor_0)
            self._t_godina_doktor_1 = self._t_godina_doktor_1 - (learning_rate * 0.005 * derivative_t_godina_doktor_1)
            self._t_godina_doktor_2 = self._t_godina_doktor_2 - (learning_rate * 0.005 * derivative_t_godina_doktor_2)

            self._t_asst_prof_0 = self._t_asst_prof_0 - (learning_rate * derivative_t_asst_prof_0)
            self._t_assoc_prof_0 = self._t_assoc_prof_0 - (learning_rate * derivative_t_assoc_prof_0)
            self._t_oblast_a_0 = self._t_oblast_a_0 - (learning_rate * derivative_t_oblast_a_0)

            # Printing the parameters for each 1000th iteration
            # print(
            #     f"Iteration {i + 1}: Cost: {current_cost:<10.5} t0: {self._t_musko_0:<10.5} t1: {self._t_god_iskustva_0:<10.5}")

        # Visualizing the weights and cost at for all iterations
        # plt.plot(weights, costs)
        # plt.title("Gradient descent")
        # plt.ylabel("Cost")
        # plt.xlabel("T0")
        # print("Current cost: " + str(current_cost))

        return self._t_musko_0, self._t_zensko_0, self._t_god_iskustva_0, self._t_godina_doktor_0, self._t_asst_prof_0, self._t_assoc_prof_0, self._t_oblast_a_0,


def remove_outliers(csv):
    outliers = []
    for i, plata in enumerate(csv.plata):
        if plata > 200000:
            outliers.append(i)

    csv.drop(index=csv.index[outliers], axis=0, inplace=True)


def split_tvt(dataset, validate=0, test=0.5):
    np.random.seed(1234)
    random = np.random.randint(1, 10000)

    validate = dataset.sample(frac=validate, random_state=random)
    dataset = dataset.drop(validate.index)

    test = dataset.sample(frac=test, random_state=random)
    dataset = dataset.drop(test.index)

    train = dataset
    return [train, validate, test]


def one_hot_data(data):
    pol_one_hot = pandas.Series(list(data.pol))
    oblast_one_hot = pandas.Series(list(data.oblast))
    zvanje_one_hot = pandas.Series(list(data.zvanje))

    pol_one_hot_data = pandas.get_dummies(pol_one_hot)
    oblast_one_hot_data = pandas.get_dummies(oblast_one_hot)
    zvanje_one_hot_data = pandas.get_dummies(zvanje_one_hot)

    if pol_one_hot_data.get("Female") is None:
        pol_one_hot_data["Female"] = 0
    if pol_one_hot_data.get("Male") is None:
        pol_one_hot_data["Male"] = 0
    if oblast_one_hot_data.get("A") is None:
        oblast_one_hot_data["A"] = 0
    if oblast_one_hot_data.get("B") is None:
        oblast_one_hot_data["B"] = 0
    if zvanje_one_hot_data.get("AssocProf") is None:
        zvanje_one_hot_data["AssocProf"] = 0
    if zvanje_one_hot_data.get("AsstProf") is None:
        zvanje_one_hot_data["AsstProf"] = 0
    if zvanje_one_hot_data.get("Prof") is None:
        zvanje_one_hot_data["Prof"] = 0

    data["Female"] = pol_one_hot_data.Female
    data["Male"] = pol_one_hot_data.Male


    data["oblast_a"] = oblast_one_hot_data.A
    data["oblast_b"] = oblast_one_hot_data.B


    data["assoc_prof"] = zvanje_one_hot_data.AssocProf
    data["asst_prof"] = zvanje_one_hot_data.AsstProf
    data["prof"] = zvanje_one_hot_data.Prof

def main(argv):
    train_file = "train.csv"
    test_file = "test_preview.csv"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]

    all_train_data = pandas.read_csv(train_file)
    test_data = pandas.read_csv(test_file)


    one_hot_data(all_train_data)
    one_hot_data(test_data)
    remove_outliers(all_train_data)

    [train_data, val, test] = split_tvt(all_train_data, validate=0.1, test=0.1)


    reg = Regression()

    reg.gradient_descent(train_data.Male, train_data.Female, train_data.godina_iskustva, train_data.godina_doktor,
    train_data.asst_prof, train_data.assoc_prof, train_data.oblast_a, train_data.plata)


    rmse = reg.root_mean_squared_error(reg.calc_function(test_data.Male, test_data.Female, test_data.godina_iskustva, test_data.godina_doktor, test_data.asst_prof, test_data.assoc_prof, test_data.oblast_a), test_data.plata)
    print(rmse)

    # print('t_musko_0 ' + str(reg._t_musko_0))
    # print('t_zensko_0 ' + str(reg._t_zensko_0))
    # print('t_god_iskustva_0 ' + str(reg._t_god_iskustva_0))
    # print('t_god_iskustva_1 ' + str(reg._t_god_iskustva_1))
    # print('t_god_iskustva_2 ' + str(reg._t_god_iskustva_2))
    # print('t_godina_doktor_0 ' + str(reg._t_godina_doktor_0))
    # print('t_godina_doktor_1 ' + str(reg._t_godina_doktor_1))
    # print('t_godina_doktor_2 ' + str(reg._t_godina_doktor_2))
    # print('t_asst_prof_0 ' + str(reg._t_asst_prof_0))
    # print('t_assoc_prof_0 ' + str(reg._t_assoc_prof_0))
    # print('t_oblast_a_0 ' + str(reg._t_oblast_a_0))

if __name__ == '__main__':
    main(sys.argv[1:])