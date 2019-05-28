# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
# https://www.kaggle.com/nholloway/the-effect-of-word-embeddings-on-bias
import numpy as np
import pickle
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras import Sequential
from keras.preprocessing import text, sequence
import keras.preprocessing.sequence
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


def get_tokenizer(sentence_list):
    """
    :param sentence_list:
    :return:
    word_index = tokenizer.word_index

    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_val = sequence.pad_sequences(x_val, maxlen=MAX_LEN)
    """
    # CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'
    # tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(sentence_list)
    return tokenizer


def load_word_embeddings(path, word_index=None, seq=' ', verbose=True):
    if word_index:
        assert isinstance(word_index, (dict, set))

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    word_embedding = {}
    total_words = 0
    load_words = 0
    with open(path) as f:
        if verbose:
            print(f"loading word embedding from path : {path} ...")
        for line in tqdm(f, disable=(not verbose)):
            word, coefs = get_coefs(*line.strip().split(seq))
            total_words += 1
            if word_index:
                if word not in word_index:
                    continue
            word_embedding[word] = coefs
            load_words += 1
    if verbose:
        embedding_dim = len(next(iter(word_embedding.values())))
        print(f"has been load {load_words} words from total {total_words} words by path : {path}")
        print(f"embedding dim : {embedding_dim}")
    return word_embedding


# create an embedding matrix for the words we have in the dataset
def build_embedding_matrix(word_index, word_embedding, lower=True, verbose=True):
    embedding_dim = len(next(iter(word_embedding.values())))
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    if verbose:
        print(f"building embedding matrix by word_index ...")
    for word, i in tqdm(word_index.items(), disable=(not verbose)):
        if lower:
            word = str(word).lower()
        embedding_vector = word_embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    if verbose:
        print(f"word_index len : {len(word_index)}")
        print(f"embedding dim : {embedding_dim}")
        print(f"embedding_matrix shape : {embedding_matrix.shape}")
    return embedding_matrix


# pip install bert_embedding
from bert_embedding import BertEmbedding
def load_word_embeddings_bert(tokenizer=None, vocab=None, save_path='../input/bert.768.pkl'):
    # %%time
    # Total CPU time (my machine): 1d 4h 7min
    if vocab is None:
        vocab = list(tokenizer.word_index.keys())
    bert_embedding = BertEmbedding()
    embedding_results = bert_embedding(vocab)
    bert_embeddings = {}
    for emb in embedding_results:
        try:
            bert_embeddings[emb[0][0]] = emb[1][0]
        except:
            pass
    with open(save_path, 'wb') as f:
        pickle.dump(bert_embeddings, f)


# https://pypi.org/project/bert-embedding/
def sentences_embedding_bert(sentence_list):
    bert_embedding = BertEmbedding()
    # test = ['मुझे चीन से प्यार है']
    # bert_embedding(test)
    result = bert_embedding(sentence_list)
    return result


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

