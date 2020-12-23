import os
import re
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from config import conf
from labels import labels as label_index


# 原始数据处理
def raw_data_process(raw_data_path, max_words_num=-1, select_word=True):
    """
    原始数据处理

    :param str raw_data_path: 原始数据地址
    :param int max_words_num: 最大词索引数
    :param bool select_word: 是否选择主干
    :return: 索引化句子数据，索引化标签，单词索引, 标签索引
    """

    print('raw_data_process: Processing Raw Data...')

    word_freq = {}
    split_words_data = []
    labels = []
    nlp_model = StanfordCoreNLP(conf["stanford_core_nlp"])
    count = 0

    index_data = []                                         # 索引化句子数据
    index_labels = []                                       # 索引化标签
    word_index = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}     # 单词索引
    # label_index                                           # 标签索引

    with open(raw_data_path, 'r') as src:
        while True:
            line1 = src.readline().split('\n')[0]
            line2 = src.readline().split('\n')[0]
            if not line1 or not line2:
                break
            # 标签
            label = line2.split('(')[0]
            # if label == 'Other':
            #     continue
            labels.append(label)
            # 句子切割成词
            sentence = line1.split("\"")[-2]
            words = []
            for word, tag in nlp_model.pos_tag(sentence.replace('%', '')):
                if select_word and not re.match(tag, 'JJ') or not select_word:
                    words.append(word)
            split_words_data.append(words)
            # 统计词频
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
            # 计数
            count += 1
            if count % 100 == 0:
                print('raw_data_process: count: ', count)

        word_freq_order = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        words_num = len(word_freq_order) if max_words_num == -1 else min(max_words_num - 3, len(word_freq_order))
        for i in range(words_num):
            word_index[word_freq_order[i][0]] = i + 3

        for sentence in split_words_data:
            index = []
            for word in sentence:
                index.append(word_index.get(word, 2))
            index_data.append(index)

        for label in labels:
            index_labels.append(label_index[label])

    print('raw_data_process: Done.')
    return index_data, index_labels, word_index, label_index


# 获得预训练编码
def get_pretrained(pretrained_data_path, word_index, embedding_len, index_depth=-1):
    """
    获得预训练编码

    :param str pretrained_data_path: 预训练编码文件的路径
    :param dict word_index: 文字索引表
    :param int embedding_len: 编码长度
    :param int index_depth: 搜索深度
    :return: 编码矩阵
    """

    count = 0

    print('get_pretrained: Indexing word vectors.')
    embeddings_index = {}  # 提取单词及其向量，保存在字典中
    with open(os.path.join(
            pretrained_data_path, 'glove.6B.' + str(embedding_len) + 'd.txt'), encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            count += 1
            if count % 100 == 0:
                print('get_pretrained: count: ', count)
            if count == index_depth:
                break

    print('get_pretrained: Found %s word vectors.' % len(embeddings_index))

    num_words = len(word_index)
    embedding_matrix = np.zeros((num_words, embedding_len))  # 词向量表
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)  # 从 GloVe 查询词向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # 写入对应位置

    return embedding_matrix


# 对于一个数字编码的句子，通过如下函数转换为字符串数据
def decode_review(text, word_index):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return ''.join([reverse_word_index.get(i, '?') + ' ' for i in text])

