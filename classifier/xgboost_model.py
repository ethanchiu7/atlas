from xgboost.sklearn import XGBClassifier
import scipy.stats as st
import feature_engineering_titanic
import best_model
import pandas as pd


if __name__ == '__main__':

    x_train, y_train, x_test = feature_engineering_titanic.read_titanic()

    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()

    # split train validate
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    # get best model
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
    xgb_clf = XGBClassifier(nthreads=-1)

    best_xgb_model = best_model.get_best_model(x_train, y_train, model=xgb_clf, params=params, n_iter=500, cv=3)

    # predict
    print("predict")
    y_predict = best_xgb_model.predict(x_test)

    # save result
    # convert nparray to dataframe
    y_predict_df = pd.DataFrame()
    y_predict_df["PassengerId"] = pd.read_csv("./data-titanic/test.csv")["PassengerId"]
    y_predict_df["Survived"] = pd.DataFrame(y_predict, columns=['Survived'])

    print(y_predict_df.head())
    # y_predict_df["Survived"] = y_predict_df["Survived"].apply(lambda x: 1 if x > 0.5 else 0)
    y_predict_df.to_csv("./data-titanic/xgb_submission.csv", index=False)
