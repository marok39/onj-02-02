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
import pickle
from sklearn.metrics import f1_score
from nltk.tokenize import RegexpTokenizer

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class_to_predict = 'Final.rating'


def clean_sentence(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    sentence = tokenizer.tokenize(sentence)
    return [w.lower() for w in sentence]


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
        self.embeddings_index = None

        # tokenizer
        self.tokenizer = None

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

    def save_tokenizer_to_file(self, tokenizer):
        # saving
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer_from_file(self):
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def string_to_vector(self, text):
        """Creates vector for cnn from passed string"""
        sequences = self.tokenizer.texts_to_sequences(text)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return data

    def create_data(self):
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        # tokenizer.fit_on_texts(self.text)
        sequences = tokenizer.texts_to_sequences(self.text)
        self.word_index = tokenizer.word_index
        print('Number of Unique Tokens', len(self.word_index))

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

        # save tokenizer
        self.save_tokenizer_to_file(tokenizer)

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
        # print(self.x_train)
        self.y_train = labels[:-nb_validation_samples]
        print('Shape of y train', self.y_train.shape)
        self.x_val = data[-nb_validation_samples:]
        print('Shape of x val', self.x_val.shape)
        self.y_val = labels[-nb_validation_samples:]
        print('Shape of y val', self.y_val.shape)

    def create_data_2(self):
        data = []

        for sentence in self.text:
            tmp = self.sentence_to_input(sentence)
            data.append(tmp)

        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)

        labels = to_categorical(np.asarray(self.labels))
        print('Shape of Data Tensor:', data.shape)
        print('Shape of Label Tensor:', labels.shape)

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        self.x_train = np.abs(data[:-nb_validation_samples])
        print('Shape of x train', self.x_train.shape)
        self.y_train = labels[:-nb_validation_samples]
        print('Shape of y train', self.y_train.shape)
        self.x_val = np.abs(data[-nb_validation_samples:])
        print('Shape of x val', self.x_val.shape)
        self.y_val = labels[-nb_validation_samples:]
        print('Shape of y val', self.y_val.shape)

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

    def create_embedding_layer_with_glove_2(self):
        embeddings_index = {}
        f = open('../data/glove.6B.100d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

        sentence = ' '.join([str(x) for x in self.text])
        arr = clean_sentence(sentence)
        word_index = set(arr)
        print('Number of Unique Tokens', len(word_index))

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for i, word in enumerate(word_index):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(word_index) + 1,
                                         EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    def create_embeddings_index(self):
        embeddings_index = {}
        f = open('../data/glove.6B.100d.txt', encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

        # saving
        with open('embeddings_index.pickle', 'wb') as handle:
            pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.embeddings_index = embeddings_index

    def sentence_to_input(self, sentence):
        arr = clean_sentence(sentence)
        embedding_matrix = np.random.random((len(arr) + 1, EMBEDDING_DIM))

        for i, word in enumerate(arr):
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # embedding_layer = Embedding(len(arr) + 1,
        #                                  EMBEDDING_DIM, weights=[embedding_matrix],
        #                                  input_length=MAX_SEQUENCE_LENGTH, trainable=True)
        return list(embedding_matrix.flatten())

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

        # self.create_data()
        # self.create_embedding_layer_with_glove()

        self.create_embeddings_index()
        self.create_data_2()
        self.create_embedding_layer_with_glove_2()

        self.train_cnn()

    def predict(self, question, response):
        """Open cnn model and make prediction"""
        # load tokenizer
        # if self.tokenizer is None:
        #     self.load_tokenizer_from_file()

        with open('embeddings_index.pickle', 'rb') as handle:
            self.embeddings_index = pickle.load(handle)

        combined_question_response = question + response

        data = []
        data.append(self.sentence_to_input(combined_question_response))
        data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
        data = np.abs(data)

        model = load_model('calculated_models/model_cnn.hdf5')
        all_classes = ["0.0", "0.5", "1.0"]

        # combined_question_response = question+response
        # data = self.string_to_vector(data)
        # print(data.shape)
        # print(data)

        result = model.predict(data)
        # print(result)
        index = np.argmax(result)
        probability = result[0][index]
        print("Predicted class %s with probability %.3f" % (all_classes[index], probability))
        return all_classes[index]


def test():
    trueScores = []
    predScores = []

    _cnn = Cnn()
    _cnn.load_tokenizer_from_file()

    df = pd.read_csv('../onj-02-02/input/Weightless_dataset_train.csv', encoding='utf8')
    i = 0
    for index, row in df.iterrows():
        i += 1
        q = row['Question']
        r = row['Response']
        true_s = int(float(row["Final.rating"].replace(",", ".")) * 10)
        trueScores.append(true_s)
        pred_s = int(float(_cnn.predict(q, r))*10)
        predScores.append(pred_s)
        if i > 5:
            break
    print("\t F1 (macro): %f" % f1_score(trueScores, predScores, average='macro'))
    print("\t F1 (micro): %f" % f1_score(trueScores, predScores, average='micro'))
    print("\t F1 (weighted): %f" % f1_score(trueScores, predScores, average='weighted'))


if __name__ == "__main__":
    cnn = Cnn()
    # cnn.create_cnn_model()
    test()

