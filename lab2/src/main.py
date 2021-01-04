from util import Dataset
from config import conf
import os, re
from Bayes import Bayes
from RNN import RNN
from labels import labels

def main():
    os.chdir(conf["WORKPATH"])
    # dataset = Dataset(conf["raw_data_train"],
    #         conf["raw_data_test"],
    #         conf["train_dataset"],
    #         conf["train_label"],
    #         conf["valid_dataset"],
    #         conf["valid_label"],
    #         conf["test_dataset"],
    #         conf["stanford_core_nlp"])
    # # dataset.run()
    # # dataset.get_word_frequence()

    # bayes = Bayes(conf["train_dataset"],
    #               conf["train_label"],
    #               conf["valid_dataset"],
    #               conf["valid_label"],
    #               conf["valid_num"],
    #               conf["result_dir"],
    #               conf["stanford_core_nlp"])
    # bayes.run()

    rnn = RNN(conf['raw_data_train'],
              conf['raw_data_test'],
              conf['result_dir'],
              conf['data_dir'],
              conf['glove_dir'],
              conf['stanford_core_nlp'],
              labels)
    rnn.run(need_process=False)
    
if __name__ == '__main__':
    main()
