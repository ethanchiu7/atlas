import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
import scipy.stats as st
import feature_engineering_titanic


def get_best_xgb_model(x, y):
    one_to_left = st.beta(10, 1)
    from_zero_positive = st.expon(0, 50)

    params = {
        "n_estimators": st.randint(3, 40),
        "max_depth": st.randint(3, 10),
        "learning_rate": st.uniform(0.05, 0.4),
        "colsample_bytree": one_to_left,
        "subsample": one_to_left,
        "gamma": st.uniform(0, 10),
        'reg_alpha': from_zero_positive,
        "min_child_weight": from_zero_positive,
    }
    clf = XGBClassifier(nthreads=-1)

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=params, n_jobs=1, n_iter=100, refit=True)
    random_search.fit(x, y)
    best_estimator = random_search.best_estimator_
    print("best_estimator")
    print(best_estimator)
    print("best_score_")
    print(random_search.best_score_)

    return best_estimator


if __name__ == '__main__':

    x_train, y_train, x_test = feature_engineering_titanic.read_titanic()

    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()

    # split train validate
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    xgb_clf = get_best_xgb_model(x_train, y_train)
