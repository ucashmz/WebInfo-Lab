import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from stanfordcorenlp import StanfordCoreNLP
from labels import labels
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from util import raw_data_process, get_pretrained
import numpy as np

class RNN:
    def __init__(self, raw_train_path, raw_test_path, result_dir, data_dir, glove_dir, stanford_core_nlp, labels):
        self.nlp_model = StanfordCoreNLP(stanford_core_nlp)
        self.raw_train_path = raw_train_path
        self.raw_test_path = raw_test_path
        self.result_dir = result_dir
        self.data_dir = data_dir
        self.glove_dir = glove_dir   # the pretrained embedding is saved here
        self.labels = labels
        self.total_words = 50000     # max words in dataset
        self.lookup_words = 300000   # max words while looking up embedding
        self.max_review_len = 30     # sentence length
        self.embedding_len = 300     # words embedding length
        self.testset_size = 400      
        self.batch_size = 500
        self.epochs = 100           # total epoches 

    def run(self, need_process=False):
        self.preprocess(need_process)
        self.train()
        self.test()


    def preprocess(self,need_process=False):
        '''
        train_index_data : words in train(and val) dataset is saved as index. Example: I like apple -> [num][num][num]
                           Each word have its own index. Sorted by frequency in train and test dataset.
        test_index_data : words in test dataset is saved as index.
        train_index_labels : labels -> index(0-9)
        word_index : dictionary for word to index
        '''
        data_tag = '_selected'  # add a tag when adj. are filtered during preprocessing

        if need_process:
            train_index_data, train_index_labels, test_index_data, word_index = raw_data_process(
                self.raw_train_path, self.raw_test_path, self.nlp_model ,max_words_num=self.total_words, cut=False, select_word=True)
            self.embedding_matrix = get_pretrained(self.glove_dir, word_index, self.embedding_len, index_depth=self.lookup_words)
            np.save(self.data_dir + 'train_index_data' + data_tag, train_index_data)
            np.save(self.data_dir + 'test_index_data' + data_tag, test_index_data)
            np.save(self.data_dir + 'train_index_labels' + data_tag, train_index_labels)
            np.save(self.data_dir + 'word_index' + data_tag, word_index)
            np.save(self.data_dir + 'embedding_matrix' + data_tag, self.embedding_matrix)
        else:
            train_index_data = np.load(self.data_dir + 'train_index_data' + data_tag + '.npy', allow_pickle=True)
            train_index_labels = np.load(self.data_dir + 'train_index_labels' + data_tag + '.npy', allow_pickle=True)
            test_index_data = np.load(self.data_dir + 'test_index_data' + data_tag + '.npy', allow_pickle=True)
            word_index = np.load(self.data_dir + 'word_index' + data_tag + '.npy', allow_pickle=True).item()
            self.embedding_matrix = np.load(self.data_dir + 'embedding_matrix' + data_tag + '.npy', allow_pickle=True)

        self.num_words = len(word_index)

        '''
        x: sentence
        y: label
        '''
        x_train = train_index_data[:-self.testset_size]
        x_val = train_index_data[-self.testset_size:]
        y_train_raw = train_index_labels[:-self.testset_size]
        y_val_raw = train_index_labels[-self.testset_size:]
        y_train = tf.one_hot(y_train_raw, depth=10)
        y_val = tf.one_hot(y_val_raw, depth=10)

        self.test_index_data = keras.preprocessing.sequence.pad_sequences(test_index_data, maxlen=self.max_review_len)
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.max_review_len)
        x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=self.max_review_len)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.train_dataset = self.train_dataset.shuffle(6000).batch(self.batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        self.val_dataset = self.val_dataset.batch(self.batch_size)
    
    def train(self):
        embedding = layers.Embedding(self.num_words, self.embedding_len, input_length=self.max_review_len, trainable=False)
        embedding.build(input_shape=(None, self.max_review_len))
        embedding.set_weights([self.embedding_matrix])

        self.model = keras.Sequential([
            embedding,
            layers.GRU(64, return_sequences=True, dropout=0.5),
            layers.GRU(64, dropout=0.5),
            # WordAttention(),
            layers.Dropout(rate=0.5),
            layers.Dense(64),
            layers.Dropout(rate=0.5),
            layers.Dense(10),
            layers.Softmax()
        ])

        self.model.build(input_shape=[None, self.max_review_len])
        self.model.summary()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        history = self.model.fit(self.train_dataset, epochs=self.epochs, validation_data=self.val_dataset)

    def test(self):
        out = self.model(self.test_index_data)
        saveTestResult(out)

    def saveTestResult(pred):
        pred_classes = tf.argmax(pred, axis=1)

        index_to_class = {}
        for key, value in labels.items():
            index_to_class[value] = key
        
        with open(self.result_dir + 'result.txt', 'w') as f:
            for i in range(len(pred_classes)):
                f.write(index_to_class[int(pred_class)] + '\n')
                # f.write(index_to_class[int(pred_classes[i])] + ", " + str(float(pred[i, pred_classes[i]])) + '\n')

    def resultProcessing(self):
        count = 0
        with open(self.result_dir + 'result.txt', 'r') as results, open(self.result_dir + 'final_result.txt', 'w') as final_result:
            while True:
                num = nums.readline()
                if not num:
                    break
                else:
                    num = int(num.split('\n')[0])
                    result = []
                    accuracy = []
                    for i in range(num):
                        line = results.readline()
                        # print(line)
                        result.append(line.split(',')[0])
                        accuracy.append(float(line.split(',')[1]))
                    id = accuracy.index(max(accuracy))
                    # print('\t', result[id], accuracy[id])
                    final_result.write(result[id] + '\n')
                    count += 1
            print(count)
        
