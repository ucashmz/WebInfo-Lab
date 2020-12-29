
from stanfordcorenlp import StanfordCoreNLP
import os, re
import numpy as np
        


class Bayes:
    def __init__(self, train_dataset, train_label, valid_dataset, valid_label, valid_num, result_dir, stanford_core_nlp):
        self.nlp_model = StanfordCoreNLP(stanford_core_nlp)
        self.train_dataset = train_dataset
        self.train_label = train_label
        self.valid_dataset = valid_dataset
        self.valid_label = valid_label
        self.result_dir = result_dir
        self.valid_num = valid_num
    
    def run(self,threshold = 3, select_word = True):
        self.load_train_data(threshold, select_word)
        self.get_result(threshold, select_word)
        self.load_result(threshold, select_word)
        self.get_accuracy(threshold, select_word)

    def get_word_frequence_for_each_label(self, select_word = True):
        print("Getting word frequency list..")
        words = [dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict()]
        with open(self.train_dataset, 'r') as data, open(self.train_label, 'r') as labels:
            sentence = data.readline()
            ctr = 0
            while sentence:
                label = int(labels.readline().split(",")[0])
                ctr += 1
                if ctr%100 == 0:
                    print(ctr)
                # print(self.nlp_model.pos_tag(sentence))
                for word, tag in self.nlp_model.pos_tag(sentence):
                    if select_word and (re.match(tag, 'NN*') or re.match(tag, 'VB*') or re.match(tag, 'IN'))  or not select_word:
                        if word not in words[label]:
                            words[label][word] = 0
                        words[label][word] += 1
                sentence = data.readline()
            
        for label in range(10):
            with open("result\\word_frequency_" + str(label) +".txt", "w") as frequence:
                result = sorted(words[label].items(), key=lambda item: item[1], reverse=True)
                for word in result:
                    frequence.write(word[0] + ", " + str(word[1]) +"\n")
        
    def load_train_data(self, threshold = 3, select_word = True):
        print("Model: Bayes")
        print("Threshold:", threshold)
        print("Select_word", select_word)

        self.dict = dict()
        with open(self.train_dataset, 'r') as data, open(self.train_label, 'r') as labels:
            sentence = data.readline()
            ctr = 0
            while sentence:
                label = int(labels.readline().split(",")[0])
                ctr += 1
                if ctr%100 == 0:
                    print(ctr)
                # print(self.nlp_model.pos_tag(sentence))
                for word, tag in self.nlp_model.pos_tag(sentence):
                    if select_word and (re.match(tag, 'NN*') or re.match(tag, 'VB*') or re.match(tag, 'IN'))  or not select_word:
                        if word not in self.dict:
                            self.dict[word] = np.array([0.0]*10)
                        self.dict[word][label] += 1
                sentence = data.readline()
        
        for word in self.dict.keys():
            len = np.sqrt(np.sum(np.square(self.dict[word])))
            if len < threshold:
                self.dict[word] = np.array([0.0]*10)
            else:
                self.dict[word] = self.dict[word]/len

    def get_result_file_name(self, threshold = 3, select_word = True):
        filename = os.path.join(self.result_dir, "bayes_result_" + str(threshold))
        if select_word: 
            filename = filename + "_selected.txt"
        else:
            filename = filename + ".txt"
        return filename

    def get_result(self, threshold = 3, select_word = True):
        np.set_printoptions(linewidth=400)

        with open(self.valid_dataset, 'r') as data, open(self.get_result_file_name(threshold, select_word), 'w') as result:
            sentence = data.readline()
            while sentence:
                sentence_vector = np.array([0.0]*10)
                for word in self.nlp_model.word_tokenize(sentence):
                    if word in self.dict:
                        sentence_vector += self.dict[word]
                result.write(str(sentence_vector) + "\n")
                sentence = data.readline()        

    def load_result(self, threshold = 3, select_word = True):
        self.result = list()
        with open(self.get_result_file_name(threshold, select_word), 'r') as result:
            sentence_vector = result.readline()
            while sentence_vector:
                sentence_vector = sentence_vector[1:-3].split()
                self.result.append(sentence_vector.index(max(sentence_vector)))
                sentence_vector = result.readline()

    def get_accuracy(self, threshold = 3, select_word = True):
        print("Threshold:", threshold)
        print("Select_word", select_word)
        correct = 0
        total = 0
        with open(self.valid_label,'r') as f:
            line = f.readline()
            while line:
                groundtruth = int(line.split(",")[0])
                if groundtruth == self.result[total]:
                    correct += 1
                total += 1
                line = f.readline()
        
        print(correct,"correct in total", total,"sentences.")