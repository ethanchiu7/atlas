#
import pandas as pd
import tensorflow as tf
import feature_engineering_titanic
from sklearn.model_selection import train_test_split
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV


class MyDnn(tf.keras.models.Sequential):

    def __init__(self, input_shape, hidden_num, layer_num):
        if hidden_num < 1:
            hidden_num = 1

        # select model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=input_shape))
        for i in range(hidden_num - 1):
            model.add(tf.keras.layers.Dense(layer_num, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

        # compile
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['accuracy'])
        self = model


def get_best_dnn_model(X, y):
    input_shape = (X.shape[1],)
    hidden_units = [st.randint(20, 100), st.randint(20, 100), st.randint(20, 100)]
    params = {
        "feature_columns": [i for i in range(input_shape[0])],
        "hidden_units": hidden_units,
        "optimizer": "rmsprop",

    }
    # model = tf.estimator.DNNClassifier(hidden_units=[64,64,64])
    random_search = RandomizedSearchCV(estimator=tf.estimator.DNNClassifier, param_distributions=params, n_iter=10, cv=3, refit=True)
    random_search.fit(X, y)

    print("best estimator : \n")
    print(random_search.best_estimator_)

    return random_search.best_estimator_


def dnn_model(input_shape=(11,)):
    # select model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    # compile
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':

    x_train, y_train, x_test = feature_engineering_titanic.read_titanic()

    x_train = x_train.as_matrix()
    y_train = y_train.as_matrix()
    x_test = x_test.as_matrix()

    # split train validate
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    # convert pd df to tf ds
    # x_train_dataset = tf.data.Dataset.from_tensors(x_train_nparray)

    dnn_model = dnn_model(input_shape=(x_train.shape[1], ))

    # dnn_model = get_best_dnn_model(x_train, y_train)

    # train
    history = dnn_model.fit(x=x_train,
                        y=y_train,
                        epochs=500,
                        batch_size=100)
    history_dict = history.history
    history_dict.keys()

    # predict
    y_predict = dnn_model.predict(x_test)

    # convert nparray to dataframe
    y_predict_df = pd.DataFrame()
    y_predict_df["PassengerId"] = pd.read_csv("./data-titanic/test.csv")["PassengerId"]
    y_predict_df["Survived"] = pd.DataFrame(y_predict, columns=['Survived'])

    y_predict_df["Survived"] = y_predict_df["Survived"].apply(lambda x: 1 if x > 0.5 else 0)

    y_predict_df.to_csv("./data-titanic/dnn_submission.csv", index=False)
