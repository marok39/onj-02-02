import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
# import os
# os.environ['KERAS_BACKEND']='theano' # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class_to_predict = 'Final.rating'


class Cnn:
    def __init__(self):
        self.df = None
        self.labels = None
        self.text = None
        self.word_index = None

        # train and test data
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        # layer
        self.embedding_layer = None

    def read_data(self, file_name='input/Weightless_dataset_train.csv'):
        """Read data from file and save it to pandas data frame"""

        df = pd.read_csv(file_name)
        df = df.dropna()
        df = df.reset_index(drop=True)
        print('Shape of dataset ', df.shape)
        print(df.columns)
        print('No. of unique classes', len(set(df[class_to_predict])))
        self.df = df
        # self.labels = df.columns

    def prepare_data(self):
        # get list of all classes
        all_classes = sorted(set(self.df[class_to_predict]))
        # convert to dict to map them
        mapped_classes = dict((note, number) for number, note in enumerate(all_classes))
        # save mapped classes to data frame
        self.df[class_to_predict] = self.df[class_to_predict].apply(lambda i: mapped_classes[i])
        # print(self.df[class_to_predict])
        text = []
        labels = []

        for i in range(self.df.shape[0]):
            q = self.df['Question'][i]
            r = self.df['Response'][i]
            # t = self.df['Text.used.to.make.inference'][i]
            # text.append(q + r + t)
            text.append(q + r)
            labels.append(self.df[class_to_predict][i])
        self.labels = labels
        self.text = text

    def string_to_vector(self, text):
        """Creates vector for cnn from passed string"""
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        self.word_index = tokenizer.word_index
        print('Number of Unique Tokens', len(self.word_index))

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return data

    def create_data(self):
        # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        # tokenizer.fit_on_texts(self.text)
        # sequences = tokenizer.texts_to_sequences(self.text)
        # word_index = tokenizer.word_index
        # print('Number of Unique Tokens', len(word_index))
        #
        # data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        data = self.string_to_vector(self.text)

        labels = to_categorical(np.asarray(self.labels))
        print('Shape of Data Tensor:', data.shape)
        print('Shape of Label Tensor:', labels.shape)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        self.x_train = data[:-nb_validation_samples]
        print('Shape of x train', self.x_train.shape)
        print(self.x_train)
        self.y_train = labels[:-nb_validation_samples]
        print('Shape of y train', self.y_train.shape)
        self.x_val = data[-nb_validation_samples:]
        self.y_val = labels[-nb_validation_samples:]

    def create_embedding_layer_with_glove(self):
        embeddings_index = {}
        f = open('../data/glove.6B.100d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))
        embedding_matrix = np.random.random((len(self.word_index) + 1, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(self.word_index) + 1,
                                         EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    def train_cnn(self, file='calculated_models/model_cnn.hdf5'):
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = self.embedding_layer(sequence_input)
        l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        l_pool2 = MaxPooling1D(5)(l_cov2)
        l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
        l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
        l_flat = Flatten()(l_pool3)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(3, activation='softmax')(l_dense)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        print("Simplified convolutional neural network")
        model.summary()
        cp = ModelCheckpoint(file, monitor='val_acc', verbose=1, save_best_only=True)

        history = model.fit(self.x_train,
                            self.y_train,
                            validation_data=(self.x_val, self.y_val),
                            epochs=15,
                            batch_size=2,
                            callbacks=[cp])

    def create_cnn_model(self):
        """
        This function creates cnn model and saves it to file
        default: calculated_models/model_cnn.hdf5.old
        """
        self.read_data()
        self.prepare_data()
        self.create_data()
        self.create_embedding_layer_with_glove()
        self.train_cnn()

    def predict(self, question, response):
        """Open cnn model and make prediction"""
        model = load_model('calculated_models/model_cnn.hdf5')
        all_classes = ["0.0", "0.5", "1.0"]

        combined_question_response = question+response
        data = self.string_to_vector(combined_question_response)

        predicted_class = model.predict(data, batch_size=1)
        index = np.argmax(predicted_class)
        probability = predicted_class[0][index]
        print("Predicted class %s with probability %.3f" % (all_classes[index], probability))
        return all_classes[index]


if __name__ == "__main__":
    cnn = Cnn()
    # cnn.create_cnn_model()

    q = 'How does Shiranna feel as the shuttle is taking off?'
    r = 'She was very scared. She was feeling the speed of the space shuttle. she remembered her mom and wanted to meet her very soon.'

    correct_result = 0.5
    cnn.predict(q, r)
