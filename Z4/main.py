import sys
import pandas

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score

import sklearn


def format_data(data, train=True):

    # data.dropna(subset=['weight'], inplace=True)
    # data.dropna(subset=['dead'], inplace=True)
    # data.dropna(subset=['airbag'], inplace=True)
    # data.dropna(subset=['seatbelt'], inplace=True)
    # data.dropna(subset=['frontal'], inplace=True)
    # data.dropna(subset=['sex'], inplace=True)
    # data.dropna(subset=['ageOFocc'], inplace=True)
    # data.dropna(subset=['yearacc'], inplace=True)
    # data.dropna(subset=['yearVeh'], inplace=True)
    # data.dropna(subset=['abcat'], inplace=True)
    # data.dropna(subset=['occRole'], inplace=True)
    # data.dropna(subset=['deploy'], inplace=True)
    # data.dropna(subset=['injSeverity'], inplace=True)

    data['weight'] = data['weight'].fillna(85)
    data['dead'] = data['dead'].fillna('alive')
    data['airbag'] = data['airbag'].fillna('airbag')
    data['seatbelt'] = data['seatbelt'].fillna('belted')
    data['frontal'] = data['frontal'].fillna(1)
    data['sex'] = data['sex'].fillna('m')
    data['ageOFocc'] = data['ageOFocc'].fillna(33)
    data['yearacc'] = data['yearacc'].fillna(1996)
    data['yearVeh'] = data['yearVeh'].fillna(1994)
    data['abcat'] = data['abcat'].fillna('unavail')
    data['occRole'] = data['occRole'].fillna('driver')
    data['deploy'] = data['deploy'].fillna(0)
    data['injSeverity'] = data['injSeverity'].fillna(2)

    data = data.reset_index()
    for index, row in data.iterrows():
        if row['injSeverity'] == 0:
            data.at[index, 'injSeverity'] = 0
        elif row['injSeverity'] == 1:
            data.at[index, 'injSeverity'] = 1
        elif row['injSeverity'] == 2:
            data.at[index, 'injSeverity'] = 2
        elif row['injSeverity'] == 3:
            data.at[index, 'injSeverity'] = 4
        elif row['injSeverity'] == 4:
            data.at[index, 'injSeverity'] = 6
        elif row['injSeverity'] == 5:
            data.at[index, 'injSeverity'] = 3
        elif row['injSeverity'] == 6:
            data.at[index, 'injSeverity'] = 5

        if row['dead'] == 'alive':
            data.at[index, 'dead'] = 0
        elif row['dead'] == 'dead':
            data.at[index, 'dead'] = 1

        if row['airbag'] == 'airbag':
            data.at[index, 'airbag'] = 1
        elif row['airbag'] == 'none':
            data.at[index, 'airbag'] = 0

        if row['seatbelt'] == 'belted':
            data.at[index, 'seatbelt'] = 1
        elif row['seatbelt'] == 'none':
            data.at[index, 'seatbelt'] = 0

        if row['sex'] == 'm':
            data.at[index, 'sex'] = 0
        elif row['sex'] == 'f':
            data.at[index, 'sex'] = 1

        if row['abcat'] == 1994:
            data.at[index, 'abcat'] = 0
        elif row['abcat'] == 'unavail':
            data.at[index, 'abcat'] = 0
        elif row['abcat'] == 'nodeploy':
            data.at[index, 'abcat'] = 1
        elif row['abcat'] == 'deploy':
            data.at[index, 'abcat'] = 2

        if row['occRole'] == 'driver':
            data.at[index, 'occRole'] = 0
        elif row['occRole'] == 'pass':
            data.at[index, 'occRole'] = 1

        # if row['speed'] == '1-9km/h': data.at[index, 'speed'] = 0.0
        # elif row['speed'] == '10-24': data.at[index, 'speed'] = 1.0
        # elif row['speed'] == '25-39': data.at[index, 'speed'] = 2.0
        # elif row['speed'] == '40-54': data.at[index, 'speed'] = 3.0
        # elif row['speed'] == '55+': data.at[index, 'speed'] = 4.0

    return data


def set_one_hot(data):
    inj_one_hot = pandas.Series(list(data.injSeverity))
    abcat_one_hot = pandas.Series(list(data.abcat))

    inj_one_hot_data = pandas.get_dummies(inj_one_hot)
    abcat_one_hot_data = pandas.get_dummies(abcat_one_hot)

    if inj_one_hot_data.get(0) is None: inj_one_hot_data[0] = 0
    if inj_one_hot_data.get(1) is None: inj_one_hot_data[1] = 0
    if inj_one_hot_data.get(2) is None: inj_one_hot_data[2] = 0
    if inj_one_hot_data.get(3) is None: inj_one_hot_data[3] = 0
    if inj_one_hot_data.get(4) is None: inj_one_hot_data[4] = 0
    if inj_one_hot_data.get(5) is None: inj_one_hot_data[5] = 0
    if inj_one_hot_data.get(6) is None: inj_one_hot_data[6] = 0

    if abcat_one_hot_data.get(0) is None: abcat_one_hot_data[0] = 0
    if abcat_one_hot_data.get(1) is None: abcat_one_hot_data[1] = 0
    if abcat_one_hot_data.get(2) is None: abcat_one_hot_data[2] = 0

    data["abcat0"] = abcat_one_hot_data[0]
    data["abcat1"] = abcat_one_hot_data[1]
    data["abcat2"] = abcat_one_hot_data[2]

    data["inj0"] = inj_one_hot_data[0]
    data["inj1"] = inj_one_hot_data[1]
    data["inj2"] = inj_one_hot_data[2]
    data["inj3"] = inj_one_hot_data[3]
    data["inj4"] = inj_one_hot_data[4]
    data["inj5"] = inj_one_hot_data[5]
    data["inj6"] = inj_one_hot_data[6]

    return data


def main(argv):
    train_file = "train.csv"
    test_file = "test_preview.csv"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]

    # train_data.describe(include="all")

    train_data = pandas.read_csv(train_file)
    test_data = pandas.read_csv(test_file)

    train_data = format_data(train_data, train=True)
    test_data = format_data(test_data, train=False)

    train_data = set_one_hot(train_data)
    test_data = set_one_hot(test_data)

    selected_columns = ['weight', 'dead', 'frontal', 'yearacc', 'yearVeh', 'abcat0', 'abcat1', 'injSeverity']

    # base_estimator = DecisionTreeClassifier(max_depth=17)
    # classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=400, learning_rate=0.125, random_state=999)
    # scores = cross_val_score(classifier, train_data[selected_columns], train_data['speed'], cv=5, scoring='f1_macro')
    # print(scores.mean())

    base_estimator = DecisionTreeClassifier(max_depth=17)
    classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=400, learning_rate=0.125, random_state=999)
    classifier.fit(train_data[selected_columns], train_data['speed'])
    result = classifier.predict(test_data[selected_columns])
    mf1 = sklearn.metrics.f1_score(test_data["speed"], result, average='macro')
    print(mf1)


if __name__ == '__main__':
    main(sys.argv[1:])