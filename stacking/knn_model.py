from sklearn.neighbors import KNeighborsClassifier
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import feature_engineering_titanic


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_best_knn_model(X, y):

    params = {
        "n_neighbors": st.randint(5, 200),
        "leaf_size": st.randint(10, 2000)
    }

    model = KNeighborsClassifier()

    random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=500, cv=3, refit=True)
    random_search.fit(X, y)

    report(random_search.cv_results_)

    print("best knn estimator : \n")
    print(random_search.best_estimator_)

    return random_search.best_estimator_


if __name__ == '__main__':
    x_train, y_train, x_test = feature_engineering_titanic.read_titanic()

    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()

    best_knn_model = get_best_knn_model(x_train, y_train)
