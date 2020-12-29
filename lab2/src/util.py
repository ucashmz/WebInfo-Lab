import os, re
from labels import labels
from stanfordcorenlp import StanfordCoreNLP

class Dataset():
    def __init__(self, 
                raw_train,
                raw_test,
                train_dataset,
                train_label,
                valid_dataset,
                valid_label,
                test_dataset,
                stanford_core_nlp,
                train_num = 6000,
                valid = True,
                valid_num = 400):

        self.raw_train = raw_train
        self.raw_test = raw_test
        self.train_dataset = train_dataset
        self.train_label = train_label
        self.valid_dataset = valid_dataset
        self.valid_label = valid_label
        self.test_dataset = test_dataset
        self.train_num = train_num
        self.valid_num = valid_num
        self.nlp_model = StanfordCoreNLP(stanford_core_nlp)
        self.labels = labels

    def run(self,
            train_num = 6000,
            valid = True,
            valid_num = 400):
        self.get_dataset(self.raw_train, self.train_dataset, self.train_label, 0, self.train_num)
        self.get_dataset(self.raw_train, self.valid_dataset, self.valid_label, self.train_num, self.train_num + self.valid_num)
        self.get_dataset(self.raw_test, self.test_dataset)

    def get_dataset(self, raw_data_path, dataset_path, label_path = None, start = 0, end = 65535):
        with open(raw_data_path , 'r') as src, open(dataset_path, 'w') as dest:
            if label_path:
                label_file = open(label_path, 'w')

            ctr = 0
            line = src.readline().split('\n')[0]
            while line and ctr < end:
                sentence = line.split("\"")[-2]
                if ctr >= start and ctr < end:
                    dest.write(sentence.replace('%', '') + "\n")
                if label_path:
                    line = src.readline()
                    label = labels[line.split("(")[0]]
                    entity0 = line.split("(")[1].split(",")[0]
                    entity1 = line.split(")")[0].split(",")[1]
                    if ctr >= start and ctr < end:
                        label_file.write(str(label) + ', ' + entity0 + ', ' + entity1 + '\n')
                ctr += 1
                line = src.readline().split('\n')[0]

            if label_path:
                label_file.close()

            print("Finished.", ctr-start, "sentences counted.")

    def get_word_frequence(self, select_word = True):
        print("Getting word frequency list..")
        words = dict()
        with open(self.train_dataset, 'r') as data, open("result\\word_frequnce.txt", 'w') as frequence:
            sentence = data.readline()
            ctr = 0
            while sentence:
                ctr += 1
                if ctr%100 == 0:
                    print(ctr)
                # print(self.nlp_model.pos_tag(sentence))
                for word, tag in self.nlp_model.pos_tag(sentence):
                    if select_word and (re.match(tag, 'NN*') or re.match(tag, 'VB*') or re.match(tag, 'IN'))  or not select_word:
                        if word not in words:
                            words[word] = 0
                        words[word] += 1
                sentence = data.readline()
            
            result = sorted(words.items(), key=lambda item: item[1], reverse=True)
            for word in result:
                frequence.write(word[0] + "\n")
            

def raw_data_process(raw_train_path, raw_test_path, nlp_model, max_words_num=-1, select_word=True, cut=False, train_size=6400):
    print('raw_data_process: Processing Raw Data...')
    word_freq = {}
    split_train_sentences = []
    split_test_sentences = []
    labels = []

    train_index_data = []
    test_index_data = []                                         
    train_index_labels = []                                       
    word_index = {'<PAD>': 0, '<START>': 1, '<UNK>': 2}     # just init with this

    with open(raw_train_path, 'r') as train_file, open(raw_test_path, 'r') as test_file:
        count = 0
        while True:
            line = test_file.readline().split('\n')[0]
            if not line:
                break
            sentence = line.split('\"')[-2]
            words = []

            noun_count = 0
            cut_sentence_count = 0
            for word, tag in nlp_model.pos_tag(sentence.replace('%', '')):
                if select_word and not re.match(tag, 'JJ') or not select_word:
                    words.append(word)
                    if cut and re.match(tag, 'NN'):
                        noun_count += 1
                        if noun_count >= 2:
                            cut_sentence_count += 1
                            split_test_sentences.append(words[:])
                            # print(words)

            if cut_sentence_count == 0:
                split_test_sentences.append(words[:])
                cut_sentence_count += 1    

            with open("test_noun_num.txt", "a") as record:
                record.write(str(cut_sentence_count) + "\n")

            count += 1
            if count % 100 == 0:
                print('raw_test_data_process: count: ', count)


        count = 0
        while True:
            line1 = train_file.readline().split('\n')[0]
            line2 = train_file.readline().split('\n')[0]
            if not line1 or not line2:
                break
            label = line2.split('(')[0]
            object1 = line2.split('(')[1].split(',')[0]
            object2 = line2.split(')')[0].split(',')[1]
            labels.append(label)
            sentence = line1.split("\"")[-2]
            words = []

            for word, tag in nlp_model.pos_tag(sentence.replace('%', '')):
                if select_word and not re.match(tag, 'JJ') or not select_word:
                    words.append(word)

            if cut and count < train_size:
                if object1 in words and object2 in words:
                    words = words[ : max(words.index(object1), words.index(object2)) + 1]

            split_train_sentences.append(words)

            count += 1
            if count % 100 == 0:
                print('raw_train_data_process: count: ', count)
        for sentence in split_train_sentences:
            for word in sentence:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        for sentence in split_test_sentences:
            for word in sentence:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        word_freq_order = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        words_num = len(word_freq_order) if max_words_num == -1 else min(max_words_num - 3, len(word_freq_order))
        for i in range(words_num):
            word_index[word_freq_order[i][0]] = i + 3

        for sentence in split_train_sentences:
            index = []
            for word in sentence:
                index.append(word_index.get(word, 2))
            train_index_data.append(index)

        for sentence in split_test_sentences:
            index = []
            for word in sentence:
                index.append(word_index.get(word, 2))
            test_index_data.append(index)

        for label in labels:
            train_index_labels.append(label_index[label])

    print('raw_data_process: Done.')

    return train_index_data, train_index_labels, test_index_data, word_index, label_index



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



            
