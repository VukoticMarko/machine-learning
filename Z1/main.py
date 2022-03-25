import matplotlib.pyplot
import numpy as np
import pandas

import matplotlib.pyplot
import sys
import matplotlib.pyplot as plt

e = 2.718281828459045

def main(argv):
    train_file = "train.csv"
    test_file = "test_preview.csv"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]


    csv = pandas.read_csv(train_file)
    test_data = pandas.read_csv(test_file)
    remove_outliers(csv)


    # plt.scatter(csv.X, csv.Y)


    gd = gradient_descent(csv.X, csv.Y)
    print(gd)

    print(gd)

    x_data = np.linspace(0,0.4,100)
    y_data = []
    result_data = []

    for x in x_data:
        y_data.append(gd[0]*(x**6) + gd[1])
        result_data.append([x, gd[0]*(x**6) + gd[1]])



    plt.plot(x_data, y_data)

    rmse = get_rmse(test_data, gd[0], gd[1])
    print("RMSE:", rmse)
    plt.show()



def get_rmse(test_data, w, b):
    predicted = []
    plt.scatter(test_data.X, test_data.Y)

    for x in test_data.X:
        predicted_value = w*(x**6) + b
        predicted.append([x, predicted_value])


    return root_mean_squared_error(test_data.values, predicted)




def remove_outliers(csv):
    outliers = []
    for i, [x,y] in enumerate(csv.values):
        if 0 < x < 1 and y > 30:
            outliers.append(i)

    csv.drop(index=csv.index[outliers], axis=0, inplace=True)


def gradient_descent(x, y, iterations=10000, learning_rate=0.2,
                     stopping_threshold=1e-6):
    # Initializing weight, bias, learning rate and iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):

        # Making predictions
        y_predicted = (current_weight * x**6) + current_bias

        # Calculationg the current cost
        current_cost = root_mean_squared_error(y, y_predicted)

        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        # Calculating the gradients
        weight_derivative = -(2 / n) * sum(6*x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i + 1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")

    # Visualizing the weights and cost at for all iterations
    # plt.figure(figsize=(8, 6))
    # plt.plot(weights, costs)
    # plt.scatter(weights, costs, marker='o', color='red')
    # plt.title("Cost vs Weights")
    # plt.ylabel("Cost")
    # plt.xlabel("Weight")
    # plt.show()
    print("Current cost: " + str(current_cost))
    return current_weight, current_bias


def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost


def root_mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    return mean_squared_error(y_true, y_predicted)


if __name__ == '__main__':
    main(sys.argv[1:])
