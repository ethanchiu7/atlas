from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import feature_engineering_titanic
import time


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

    # report(random_search.cv_results_)

    print("best estimator : \n")
    # print(random_search.best_estimator_)
    print("return random_search")

    return random_search


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

    best_knn_model = get_best_model(x_train, y_train, model=knn, params=knn_params, n_iter=10, cv=3)

    # predict
    print("x_test.shape: " + x_test.shape)
    print(x_test.shape)
    print(time.time())
    y_predict = best_knn_model.predict(x_test)
    print(time.time())
    print(y_predict)
