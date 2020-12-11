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
                    dest.write(sentence + "\n")
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

    def get_word_frequnce(self, select_word = True):
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
            



            
