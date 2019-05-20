# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout, Activation
stop_words = stopwords.words('english')


def get_word_count(sentences, verbose=True):
    """
    :param verbose:
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    word_count = {}
    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                word_count[word] += 1
            except KeyError:
                word_count[word] = 1
    return word_count


def get_word_index(word_count, num_words=None, verbose=False):
    word_index = {}
    word_index = sorted(word_count.items(), key=lambda kv: kv[1])[::-1][:num_words]
    word_index = {value[0]: index for index, value in enumerate(word_index)}
    index_word = {word_index[word]: word for word in word_index}
    if verbose:
        print("len of word_index :{}".format(len(word_index)))
    return word_index, index_word


# load the GloVe vectors in a dictionary
def get_glove_vectors(file_path='glove.840B.300d.txt', verbose=0):
    word_embedding = {}
    f = open(file_path)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embedding[word] = coefs
    f.close()
    if verbose > 0:
        print('Found %s word vectors.' % len(word_embedding))
    return word_embedding


# create an embedding matrix for the words we have in the dataset
def get_embedding_matrix(word_index, word_embedding, embedding_dim=300):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in tqdm(word_index.items()):
        embedding_vector = word_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# A simple LSTM with glove embeddings and two dense layers
def get_lstm_model_with_embedding_matrix(word_index, embedding_dim=300, input_length=70, embedding_matrix=None):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=input_length,
                        trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))

    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# this function creates a normalized vector for the whole sentence
def sent2vec(sentence, word_embedding):
    # words = str(s).lower().decode('utf-8')
    words = str(sentence).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(word_embedding[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

