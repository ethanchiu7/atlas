from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import feature_engineering_titanic
import time
import pandas as pd
import sys
import os


def report(results, n_top=3):
    """
    Utility function to report best scores
    :param results: RandomizedSearchCV.cv_results_
    :param n_top: best score model num
    :return:
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_best_model(X, y, model, params, n_iter=100, cv=3):
    print("select model by CV ...")

    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=n_iter, cv=cv, refit=True)
    random_search.fit(X, y)

    report(random_search.cv_results_)

    print("best estimator : \n")
    print(random_search.best_estimator_)

    return random_search.best_estimator_


def save_csv(y, save_path):
    # convert nparray to dataframe
    y_predict_df = pd.DataFrame()
    y_predict_df["PassengerId"] = pd.read_csv("./data-titanic/test.csv")["PassengerId"]
    y_predict_df["Survived"] = pd.DataFrame(y, columns=['Survived'])

    print(y_predict_df.head())
    # y_predict_df["Survived"] = y_predict_df["Survived"].apply(lambda x: 1 if x > 0.5 else 0)
    y_predict_df.to_csv(save_path, index=False)
    print("predict result has been save : {}".format(save_path))


if __name__ == '__main__':
    x_train, y_train, x_test = feature_engineering_titanic.read_titanic()

    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()

    # get best model
    knn = KNeighborsClassifier()
    knn_params = {
        "n_neighbors": st.randint(5, 200),
        "leaf_size": st.randint(10, 2000)
    }

    best_estimator = get_best_model(x_train, y_train, model=knn, params=knn_params, n_iter=1000, cv=3)

    # predict
    print("predict")
    y_predict = best_estimator.predict(x_test)

    # save result
    knn_save_path = os.path.join(os.getcwd(), "data-titanic", "knn_submission.csv")
    save_csv(y_predict, knn_save_path)
