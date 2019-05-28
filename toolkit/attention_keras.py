# https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold/#data
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_sequences=False, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.return_sequences = return_sequences
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(
                                K.reshape(x, (-1, features_dim)),
                                K.reshape(self.W, (features_dim, 1))
                              ),
                        (-1, step_dim)
                        )

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        step_weight = K.exp(eij)

        if mask is not None:
            step_weight *= K.cast(mask, K.floatx())

        step_weight /= K.cast(K.sum(step_weight, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # dim from (step_weight) to (step_weight, 1)
        step_weight = K.expand_dims(step_weight)
        # element-wise product
        weighted_input = x * step_weight
        if self.return_sequences:
            return weighted_input
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


import keras.layers as L
from keras.models import Model
from keras.optimizers import Adam


def build_model_attention(verbose=False, compile=True, maxlen=200, embedding_matrix=None):
    sequence_input = L.Input(shape=(maxlen,), dtype='int32')
    embedding_layer = L.Embedding(*embedding_matrix.shape,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    x = embedding_layer(sequence_input)
    x = L.SpatialDropout1D(0.2)(x)
    x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)

    att = Attention(maxlen)(x)
    avg_pool1 = L.GlobalAveragePooling1D()(x)
    max_pool1 = L.GlobalMaxPooling1D()(x)

    x = L.concatenate([att,avg_pool1, max_pool1])

    preds = L.Dense(1, activation='sigmoid')(x)


    model = Model(sequence_input, preds)
    if verbose:
        model.summary()
    if compile:
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.005), metrics=['acc'])
    return model