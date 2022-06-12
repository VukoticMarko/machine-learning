import sys
# import matplotlib.pyplot as plt
import pandas
import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


def format_data(data):

    data.dropna(subset=['year'], inplace=True)
    data.dropna(subset=['age'], inplace=True)
    data.dropna(subset=['maritl'], inplace=True)
    data.dropna(subset=['race'], inplace=True)
    data.dropna(subset=['education'], inplace=True)
    data.dropna(subset=['jobclass'], inplace=True)
    data.dropna(subset=['health'], inplace=True)
    data.dropna(subset=['health_ins'], inplace=True)
    data.dropna(subset=['wage'], inplace=True)

    data = data.reset_index()

    return data


def set_one_hot (data):
    maritl_one_hot = pandas.Series(list(data.maritl))
    education_one_hot = pandas.Series(list(data.education))
    jobclass_one_hot = pandas.Series(list(data.jobclass))
    health_one_hot = pandas.Series(list(data.health))
    healthins_one_hot = pandas.Series(list(data.health_ins))
    race_one_hot = pandas.Series(list(data.race))

    maritl_one_hot_data = pandas.get_dummies(maritl_one_hot)
    education_one_hot_data = pandas.get_dummies(education_one_hot)
    jobclass_one_hot_data = pandas.get_dummies(jobclass_one_hot)
    health_one_hot_data = pandas.get_dummies(health_one_hot)
    healthins_one_hot_data = pandas.get_dummies(healthins_one_hot)
    race_one_hot_data = pandas.get_dummies(race_one_hot)

    if maritl_one_hot_data.get('1. Never Married') is None: maritl_one_hot_data['1. Never Married'] = 0
    if maritl_one_hot_data.get('2. Married') is None: maritl_one_hot_data['2. Married'] = 0
    if maritl_one_hot_data.get('3. Widowed') is None: maritl_one_hot_data['3. Widowed'] = 0
    if maritl_one_hot_data.get('4. Divorced') is None: maritl_one_hot_data['4. Divorced'] = 0
    if maritl_one_hot_data.get('5. Separated') is None: maritl_one_hot_data['5. Separated'] = 0

    if education_one_hot_data.get('1. < HS Grad') is None: education_one_hot_data['1. < HS Grad'] = 0
    if education_one_hot_data.get('2. HS Grad') is None: education_one_hot_data['2. HS Grad'] = 0
    if education_one_hot_data.get('3. Some College') is None: education_one_hot_data['3. Some College'] = 0
    if education_one_hot_data.get('4. College Grad') is None: education_one_hot_data['4. College Grad'] = 0
    if education_one_hot_data.get('5. Advanced Degree') is None: education_one_hot_data['5. Advanced Degree'] = 0

    if jobclass_one_hot_data.get('1. Industrial') is None: jobclass_one_hot_data['1. Industrial'] = 0
    if jobclass_one_hot_data.get('2. Information') is None: jobclass_one_hot_data['2. Information'] = 0

    if health_one_hot_data.get('1. <=Good') is None: health_one_hot_data['1. <=Good'] = 0
    if health_one_hot_data.get('2. >=Very Good') is None: health_one_hot_data['2. >=Very Good'] = 0

    if healthins_one_hot_data.get('1. Yes') is None: healthins_one_hot_data['1. Yes'] = 0
    if healthins_one_hot_data.get('2. No') is None: healthins_one_hot_data['2. No'] = 0

    if race_one_hot_data.get('1. White') is None: race_one_hot_data['1. White'] = 0
    if race_one_hot_data.get('2. Black') is None: race_one_hot_data['2. Black'] = 0
    if race_one_hot_data.get('3. Asian') is None: race_one_hot_data['3. Asian'] = 0
    if race_one_hot_data.get('4. Other') is None: race_one_hot_data['4. Other'] = 0

    


    data["maritl1"] = maritl_one_hot_data['1. Never Married']
    data["maritl2"] = maritl_one_hot_data['2. Married']
    data["maritl3"] = maritl_one_hot_data['3. Widowed']
    data["maritl4"] = maritl_one_hot_data['4. Divorced']
    data["maritl5"] = maritl_one_hot_data['5. Separated']

    data["education1"] = education_one_hot_data['1. < HS Grad']
    data["education2"] = education_one_hot_data['2. HS Grad']
    data["education3"] = education_one_hot_data['3. Some College']
    data["education4"] = education_one_hot_data['4. College Grad']
    data["education5"] = education_one_hot_data['5. Advanced Degree']

    data["jobclass1"] = jobclass_one_hot_data['1. Industrial']
    data["jobclass2"] = jobclass_one_hot_data['2. Information']


    data["health1"] = health_one_hot_data['1. <=Good']
    data["health2"] = health_one_hot_data['2. >=Very Good']
    
    data["healthins1"] = healthins_one_hot_data['1. Yes']
    data["healthins2"] = healthins_one_hot_data['2. No']
    

    data["race1"] = race_one_hot_data['1. White']
    data["race2"] = race_one_hot_data['2. Black']
    data["race3"] = race_one_hot_data['3. Asian']
    data["race4"] = race_one_hot_data['4. Other']


    return data


def grid_search(trainX, trainY):
    # https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html

    pca = PCA()
    decision_tree = DecisionTreeClassifier()

    pipe = Pipeline(steps=[("pca", pca), ("decision_tree", decision_tree)])
    param_grid = {
        "pca__n_components": [1,2,3,4,5, 6, 7, 8, 9, 10],
        "decision_tree__max_depth": [1,5,10,20,30,40,50,60,70,80,90,100],
    }

    search = GridSearchCV(pipe, param_grid, n_jobs=8)
    search.fit(trainX, trainY)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)


def main(argv):
    train_file = "train.csv"
    test_file = "test_preview.csv"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]

    # train_data.describe(include="all")

    train_data = pandas.read_csv(train_file)

    train_data = format_data(train_data)
    train_data = set_one_hot(train_data)

    selected_columns = ['year','age', 'maritl1', 'maritl2', 'maritl3', 'maritl4', 'education1', 'education2', 'education3', 'education4', 'jobclass1', 'health1', 'healthins1']


    test_data = pandas.read_csv(test_file)
    test_data = format_data(test_data)
    test_data = set_one_hot(test_data)

    # grid_search(train_data[selected_columns], train_data['race'])

    pca = PCA(n_components=11)
    pca_train = pca.fit_transform(train_data[selected_columns])
    decision_tree = DecisionTreeClassifier(max_depth=100, random_state=0)
    decision_tree.fit(pca_train, train_data['race'])

    pca_test = pca.transform(test_data[selected_columns])
    result = decision_tree.predict(pca_test)
    score = sklearn.metrics.f1_score(test_data["race"], result, average='macro')
    print(score)

    # scores = cross_val_score(decision_tree, pca_train, train_data['race'], cv=5, scoring='f1_macro')
    # print(scores.mean())

if __name__ == '__main__':
    main(sys.argv[1:])