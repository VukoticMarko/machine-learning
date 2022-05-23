import sys
import pandas
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.svm import SVC
import re

STOP_WORDS = text.ENGLISH_STOP_WORDS

class ClickBaiter():
    vectorizer = TfidfVectorizer(stop_words=[], lowercase=False, token_pattern=r"(?u)([A-Za-z]+|[^a-zA-Z\d\s]+|[0-9])")
    svc = SVC(kernel='linear', C=1.4)

    def create_word_vector(self, base_data):
        return self.vectorizer.fit_transform(base_data)

    def covert_to_vector(self, test_data):
        return self.vectorizer.transform(test_data)

    def fit(self, data, result):
        self.svc.fit(data, result)

    def predict(self, data):
        return self.svc.predict(data)


def replace_number_with_same(dataframe):
    for index in range(len(dataframe)):
        dataframe.at[index, 'text'] = re.sub(r'\d+', '999', dataframe.at[index, 'text'])


def split_tvt(dataset, validate=0, test=0.5):
    np.random.seed(1000)
    random = np.random.randint(1, 10000)

    validate = dataset.sample(frac=validate, random_state=random)
    dataset = dataset.drop(validate.index)

    test = dataset.sample(frac=test, random_state=random)
    dataset = dataset.drop(test.index)

    train = dataset

    train = train.reset_index()
    validate = validate.reset_index()
    test = test.reset_index()

    return [train, validate, test]


def main(argv):
    train_file = "train.json"
    test_file = "test_preview.json"
    if len(argv) >= 2:
        train_file = argv[0]
        test_file = argv[1]

    train_data = pandas.read_json(train_file)
    test_data = pandas.read_json(test_file)

    # train_data, test, test_data = split_tvt(train_data, validate=0, test=0.2)

    replace_number_with_same(train_data)
    replace_number_with_same(test_data)

    cb = ClickBaiter()
    word_vector = cb.create_word_vector(train_data["text"])
    word_vector_test = cb.covert_to_vector(test_data["text"])

    cb.fit(word_vector, train_data["clickbait"])

    result = cb.predict(word_vector_test)

    # for i,res in enumerate(result):
    #     if test_data.at[i, 'clickbait'] != res:
    #         print(test_data.at[i, 'text'])

    mf1 = sklearn.metrics.f1_score(test_data["clickbait"], result, average='micro')
    print(mf1)


if __name__ == '__main__':
    main(sys.argv[1:])