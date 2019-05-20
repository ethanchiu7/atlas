import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
from sklearn.preprocessing import LabelEncoder

# TensorFlow Dataset pipeline
train_df = pd.read_csv("./data-titanic/train.csv")
test_df = pd.read_csv("./data-titanic/test.csv")

# Label
labels_train = train_df['Survived']

# Train features
userId_train = train_df['PassengerId']
pclass_train = train_df['Pclass']
sex_train = train_df['Sex']
age_train = train_df['Age']

# Test features
userId_test = test_df['PassengerId']
pclass_test = test_df['Pclass']
sex_test = test_df['Sex']
age_test = test_df['Age']

# sparse features

sex_encoder = LabelEncoder()
sex_encoder.fit(sex_train)
sex_train = sex_encoder.transform(sex_train)
sex_test = sex_encoder.transform(sex_test)
sex_num_class = np.max(sex_train) + 1

# convert labels to one hot
pclass_train = keras.utils.to_categorical(pclass_train, 5)
pclass_test = keras.utils.to_categorical(pclass_test, 5)

sex_train = keras.utils.to_categorical(sex_train, sex_num_class)
sex_test = keras.utils.to_categorical(sex_test, num_classes=sex_num_class)

# age_train = keras.utils.to_categorical(age_train, 100)
# age_test = keras.utils.to_categorical(age_test, 100)

# wide model
pclass_input = layers.Input(shape=(5, ))
sex_input = layers.Input(shape=(sex_num_class, ))
# age_input = layers.Input(shape=(100, ))

merged_layers = layers.concatenate([pclass_input, sex_input])
merged_layers = layers.Dense(256, activation='relu')(merged_layers)
predictions = layers.Dense(1)(merged_layers)
wide_model = keras.Model(inputs=[pclass_input, sex_input], outputs=predictions)
print(wide_model.summary())

# deep model
deep_input = layers.Input(shape=(1, ))
embedding = layers.Embedding(10000, 32)(deep_input)
embedding = layers.Flatten()(embedding)
embed_out = layers.Dense(1)(embedding)
deep_model = keras.Model(inputs=deep_input, outputs=embed_out)
print(deep_model.summary())

# Combine wide and deep into one model
merged_output = layers.concatenate([wide_model.output, deep_model.output])
merged_output = layers.Dense(1)(merged_output)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_output)
print(combined_model.summary())

combined_model.compile(loss=losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

combined_model.fit(x=[pclass_train, sex_train, userId_train], y=labels_train, epochs=10, batch_size=128)

