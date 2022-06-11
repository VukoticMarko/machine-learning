import sys
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import sklearn

# Disable warnings from scikit
# import warnings
#
#
# def warn(*args, **kwargs):
#     pass
#
#
# warnings.warn = warn


def format_data(data, train=True):
    if train:
        data.dropna(subset=['income'], inplace=True)
        data.dropna(subset=['infant'], inplace=True)
        data.dropna(subset=['region'], inplace=True)
        data.dropna(subset=['oil'], inplace=True)
    else:
        data['income'] = data['income'].fillna(969)
        data['infant'] = data['infant'].fillna(92)
        data['region'] = data['region'].fillna('Africa')
        data['oil'] = data['oil'].fillna('no')

    for index, row in data.iterrows():
        if row['region'] == 'Africa':
            data.at[index, 'region'] = 0
        elif row['region'] == 'Americas':
            data.at[index, 'region'] = 1
        elif row['region'] == 'Asia':
            data.at[index, 'region'] = 2
        elif row['region'] == 'Europe':
            data.at[index, 'region'] = 4

    data = data.reset_index()
    return data


def set_one_hot(data):
    region_one_hot = pandas.Series(list(data.region))
    oil_one_hot = pandas.Series(list(data.oil))

    region_one_hot_data = pandas.get_dummies(region_one_hot)
    oil_one_hot_data = pandas.get_dummies(oil_one_hot)

    if region_one_hot_data.get('Africa') is None: region_one_hot_data['Africa'] = 0
    if region_one_hot_data.get('Americas') is None: region_one_hot_data['Americas'] = 0
    if region_one_hot_data.get('Asia') is None: region_one_hot_data['Asia'] = 0
    if region_one_hot_data.get('Europe') is None: region_one_hot_data['Europe'] = 0

    if oil_one_hot_data.get('no') is None: oil_one_hot_data[0] = 0
    if oil_one_hot_data.get('yes') is None: oil_one_hot_data[1] = 0

    data["oily"] = oil_one_hot_data['no']
    data["oiln"] = oil_one_hot_data['yes']

    data["reg0"] = region_one_hot_data['Africa']
    data["reg1"] = region_one_hot_data['Americas']
    data["reg2"] = region_one_hot_data['Asia']
    data["reg3"] = region_one_hot_data['Europe']

    return data


def remove_outliers(data):
    outliers = []
    for index, row in data.iterrows():
        outlier = False
        if row.region == "Africa" and row.income > 3000: outlier = True
        # if row.region == "Americas" and row.income > 4000: outlier = True
        # if row.region == "Americas" and row.income > 3000: outlier = True
        if row.region == "Asia" and row.infant >= 300: outlier = True
        # if row.infant == 300: outlier = True

        if outlier:
            outliers.append(index)

    data.drop(index=data.index[outliers], axis=0, inplace=True)

    data = data.reset_index()
    return data


def split_tvt(dataset, validate=0, test=0.4):
    random = 400
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

    # train_data.describe(include="all")

    train_data = pandas.read_csv(train_file)
    train_data = remove_outliers(train_data)
    train_data = format_data(train_data)
    train_data = set_one_hot(train_data)

    selected_columns = ['income', 'infant', 'oily']

    best_random_state = 90
    gaus = GaussianMixture(n_components=4,
                           covariance_type='diag',
                           tol=0.0000001,
                           reg_covar=1e-8,
                           max_iter=10000,
                           n_init=1,
                           init_params='random')

    # maxscore = 0
    # best_random_state = 0
    # train, validate_data, test_data = split_tvt(train_data, validate=0, test=0.3)
    #
    # for i in range(100):
    #     gaus.random_state = i
    #
    #     gaus.fit(train_data[selected_columns], train_data['region'])
    #
    #     result = gaus.predict(test_data[selected_columns])
    #     score = sklearn.metrics.v_measure_score(test_data['region'].to_list(), result)
    #     if score >= maxscore:
    #         maxscore = score
    #         best_random_state = i
    #
    # print("Best index: " + str(best_random_state))
    # print("Score: " + str(maxscore))

    gaus.random_state = best_random_state

    gaus.fit(train_data[selected_columns], train_data['region'])

    test_data = pandas.read_csv(test_file)
    test_data = format_data(test_data, train=False)
    test_data = set_one_hot(test_data)

    result = gaus.predict(test_data[selected_columns])
    score = sklearn.metrics.v_measure_score(test_data['region'].to_list(), result)
    print(score)


if __name__ == '__main__':
    main(sys.argv[1:])