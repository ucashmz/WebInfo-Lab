import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow_temp.utils import raw_data_process, get_pretrained, decode_review
from tensorflow_temp.attention import WordAttention
import numpy as np

# 全局定义
total_words = 50000     # 词汇表大小 N_vocab
max_review_len = 30     # 句子最大长度 s，大于的句子部分将截断，小于的将填充
embedding_len = 100     # 词向量特征长度 n
testset_size = 1000     # 测试集大小
batch_size = 200        # batch 大小

# 预计算控制
need_precess = False
data_path = './data/'
data_tag = '_selected'

# 预计算储存与读取
if need_precess:
    index_data, index_labels, word_index, label_index = raw_data_process(
        '../../raw_data/train.txt', max_words_num=total_words, select_word=True)
    embedding_matrix = get_pretrained('D:\GloVe', word_index, embedding_len, index_depth=100000)
    np.save(data_path + 'index_data' + data_tag, index_data)
    np.save(data_path + 'index_labels' + data_tag, index_labels)
    np.save(data_path + 'word_index' + data_tag, word_index)
    np.save(data_path + 'label_index' + data_tag, label_index)
    np.save(data_path + 'embedding_matrix' + data_tag, embedding_matrix)
else:
    index_data = np.load(data_path + 'index_data' + data_tag + '.npy', allow_pickle=True)
    index_labels = np.load(data_path + 'index_labels' + data_tag + '.npy', allow_pickle=True)
    word_index = np.load(data_path + 'word_index' + data_tag + '.npy', allow_pickle=True).item()
    label_index = np.load(data_path + 'label_index' + data_tag + '.npy', allow_pickle=True).item()
    embedding_matrix = np.load(data_path + 'embedding_matrix' + data_tag + '.npy', allow_pickle=True)

# print(index_data, index_labels, word_index, label_index)

num_words = len(word_index)

# 分割训练集和测试集
state = np.random.get_state()
np.random.shuffle(index_data)
np.random.set_state(state)
np.random.shuffle(index_labels)

x_train = index_data[:-testset_size]
x_test = index_data[-testset_size:]
y_train_raw = index_labels[:-testset_size]
y_test_raw = index_labels[-testset_size:]
y_train = tf.one_hot(y_train_raw, depth=10)
y_test = tf.one_hot(y_test_raw, depth=10)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(6000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# 网络结构
embedding = layers.Embedding(num_words, embedding_len, input_length=max_review_len, trainable=False)
embedding.build(input_shape=(None, max_review_len))
embedding.set_weights([embedding_matrix])

model = keras.Sequential([
    embedding,
    layers.GRU(64, return_sequences=True, dropout=0.5),
    # layers.GRU(64, dropout=0.5),
    WordAttention(),
    layers.Dropout(rate=0.5),
    layers.Dense(64),
    layers.Dropout(rate=0.5),
    layers.Dense(10),
    layers.Softmax()
])

# 网络训练
model.build(input_shape=[None, max_review_len])
model.summary()
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
history = model.fit(train_dataset, epochs=100, validation_data=test_dataset)


# 测试
def test(label, pred):
    predict_count = np.zeros(10)
    correct_count = np.zeros(10)
    pred = tf.sigmoid(pred)
    pred_class = tf.argmax(pred, axis=1)
    for i in range(len(label)):
        predict_count[label[i]] += 1
        if pred[i, pred_class[i]] > 0.5:
            if pred_class[i] == label[i]:
                correct_count[label[i]] += 1
        else:
            if 9 == label[i]:
                correct_count[label[i]] += 1
    return predict_count, correct_count

# out = model(x_test)
# predict_count, correct_count = test(y_test_raw, out)
# predict_count = predict_count[:-1]
# correct_count = correct_count[:-1]
#
# print('Correctness Info:')
# print(predict_count)
# print(correct_count / predict_count)
# print(np.sum(correct_count) / np.sum(predict_count))

for i in range(len(y_test_raw)):
    if y_test_raw[i] == 0:
        out = model(tf.reshape(x_test[i], shape=(1, -1)), training=False)
        print(decode_review(x_test[i], word_index))
        print(out)
        print(y_test_raw[i])
        break
