# 手工特征工程
# 数据来源 Kaggle Titanic : https://www.kaggle.com/c/titanic

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def get_titanic_fea(dataset):
    """
    get data-titanic feature engineering result
    :param dataset: pandas.DataFrame
    :return:
    """
    dataset['Name_length'] = dataset['Name'].apply(len)

    # Mapping Sex 不在map定义的 就是NaN
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # [Embarked]
    dataset['Embarked'] = dataset['Embarked'].fillna('0')
    dataset['Fare'] = dataset['Fare'].fillna(0)
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'0': 0, 'S': 1, 'C': 2, 'Q': 3}).astype(int)

    # [Fare]
    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4)
    # Mapping Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # [Age]
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)
    # Mapping Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    # [Name]
    # 称谓 Mr 、Miss 等
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
    dataset['Title'] = dataset['Name'].apply(get_title)

    # 只保留4类Title
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    dataset = dataset.drop(drop_elements, axis=1)
    dataset = dataset.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

    return dataset


def split_xy(dataset, label_key):

    return dataset[label_key], dataset.drop(columns=label_key)


def read_titanic():
    train_data = pd.read_csv("./data-titanic/train.csv")
    test_data = pd.read_csv("./data-titanic/test.csv")

    train_data = train_data.dropna(axis=0, how="all")
    train_data = get_titanic_fea(train_data)
    test_data = get_titanic_fea(test_data)

    y_train, x_train = split_xy(train_data, "Survived")
    x_test = test_data

    return x_train, y_train, x_test


if __name__ == '__main__':
    x_train, y_train, x_test = read_titanic()

    print("x_train.shap : {}\n".format(x_train.shape))
    print(x_train.head())
    print("y_train.shap : {}\n".format(y_train.shape))
    print(y_train.head())
    print('x_test.shap : {}\n'.format(x_test.shape))
    print(x_test.head())


