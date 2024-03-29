import csv
import sys
import os
import re
import linecache
import time
import math
import numpy as np
from config import conf
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

'''
Used to read mail data. return paragraphs without heading
'''


def read_file(file_path, error_log, counter):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
        except:
            with open(error_log, 'a')as log:
                log.write(str(counter) + ",problems in " + file_path + "\n")
            print("problems in", file_path)
            return []
        paragraphs = content.split('\n\n')[1:]  # discard header of the email
    return paragraphs

'''
Get words in a paragraph, No numbers, punctuations, stopwords in words.
Cache (type:dict()) is used to reduce time cost during getting word's stem.
Cache[original-word] = word's-stem
Return a list of words
'''


def tokenize_paragraph(paragraph, stops, cache):
    cut_word_without_num = re.split(
        r'[\t\n !\"#$%&\'()*+,-./0123456789:;<=>?@\[\]_~]', paragraph.lower())
    word_without_stops = [
        word for word in cut_word_without_num if word not in stops and len(word) > 1]
    # word_without_num = [word for word in word_without_stops if not re.match('.*[0-9].*',word)]
    word_stem = list()
    for word in word_without_stops:
        if word not in cache:
            cache[word] = PorterStemmer().stem(word)
        word_stem.append(cache[word])
        # word_stem.append(PorterStemmer().stem(word))
    return word_stem


'''
Define some stopwords
Cache used for tokenize_paragraph().
Return a list of words occur in the file
'''


def tokenize_file(file_path, error_log, counter, cache):
    paragraphs = read_file(file_path, error_log, counter)
    stops = set(stopwords.words(
        "english")+['am', 'pm', 'ect', 'cc', 'ps', 'www', 'com', 're'])  # some stopwords added
    # print(stops)
    words = list()
    for paragraph in paragraphs:
        result = tokenize_paragraph(paragraph, stops, cache)
        words += result
    # print(words)
    return words


'''
Limit: a set of words that need to be considered. For example, first 1000 words most frequently occured in all files.
Cache used for tokenize_paragraph().
Words_tf[word] = tf_value of the word
Return a dictionary
'''


def get_tf(file_path, error_log, counter, cache, limit):
    words_in_file = tokenize_file(file_path, error_log, counter, cache)
    words_tf = dict.fromkeys(limit, 0)
    for word in words_in_file:
        if word in limit:
            words_tf[word] += 1
    for word in limit:
        if words_tf[word] > 0:
            words_tf[word] = float('%.03f' % (
                1 + math.log(words_tf[word], 10)))
    return words_tf





class DataLoader:
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.filename_path = config['filename_path']
        self.inverted_table = config['inverted_table']
        self.tf_table = config['tf_table']
        self.checkpoint = config['checkpoint']
        self.error_log = config['error_log']
        self.words_list = config['words_list']
        self.tfidf_table_path = config['tfidf_table_path']

    '''
    Do initialization.
    These files will exist after initialization:
        filename        - All files' name saved in the file. Inverted table and tfidf table are based on this file.
        words_list      - A file with first 1000 words most frequently occured in all files. Organized as "words, occur-times". Sorted.
                          tfidf vector is calculated base on this file.
        inverted_table  - Merged by small fragments(checkpoints).
        tfidf_table     - Merged by small tf-table fragments(checkpoints). Need inverted_table to calculate tfidf vector.
    '''

    def init(self):
        if not (os.path.exists(self.filename_path) and os.path.exists(self.words_list)):
            if os.path.exists(self.inverted_table + '.csv') or os.path.exists(self.tfidf_table_path):# or self.get_inverted_table_list or self.get_tf_table_list:
                print(
                    "ERROR: Without filename or words_list, but inverted_table or tfidf_table found.")
                print("       Remove inverted_table and tfidf_table first and restart.")
                return
            else:
                if not os.path.exists(self.filename_path):
                    print("Get filenames list")
                    self.save_filename()
                if not os.path.exists(self.words_list):
                    print(
                        "Counting words.. Need to get a list of first 1000 of most frequently occured words")
                    self.words_counting()
        print("filename and words list found.")

        self.file_num = self.get_file_num()
        self.words_list_sorted = self.load_words_list()

        if not os.path.exists(self.inverted_table + '.csv'):
            if not os.path.exists(self.inverted_table + str(self.file_num) + '.csv'):
                # Generate small fragments of inverted_table and tf_table
                self.get_inverted_table_and_tf_table()
            self.merge_inverted_table()

        self.word_occurence = self.get_word_occurence()  # Used to calculate tfidf vector

        if not os.path.exists(self.tfidf_table_path):
            if not os.path.exists(self.tf_table + str(self.file_num) + '.csv'):
                # Generate small fragments of inverted_table and tf_table
                self.get_inverted_table_and_tf_table()
            self.get_tfidf_table()
        print("Initialization Complete.")

    '''
    For inverted_table and tf_table 's generation.
    Fragments saved every $checkpoint files.
    This function is used to get where to continue the process, return an int.
    '''

    def load_checkpoint(self):
        i = 0
        while (os.path.exists(self.inverted_table + str(i + self.checkpoint) + ".csv") and os.path.exists(self.tf_table + str(i + self.checkpoint) + ".csv")):
            i += self.checkpoint
        if i != 0:
            print("Load from checkpoint", i)
        else:
            print("Start without checkpoint.")
        return i

    '''
    To get words in words_list.
    Return a list, in which words are sorted by times they occur in all files.
    tfidf vector is calculated base on this list.
    '''

    def load_words_list(self):
        words = list()
        with open(self.words_list, 'r') as f:
            table_line = f.readline().split('\n')[0]
            while table_line:
                words.append(table_line.split(",")[0])
                table_line = f.readline().split('\n')[0]
        # print(words)
        return words

    def get_inverted_table_and_tf_table(self):
        cache = dict()  # Used to speed up tokenize process
        # To reduce finding time, use set to save words list here. The set is NOT sorted
        words_list = set(self.words_list_sorted)
        checkpoint = self.load_checkpoint()
        t_start = time.process_time()
        with open(self.filename_path, 'r') as f:
            file_ctr = 0
            file_operation_ctr = 0
            filename = f.readline().split('\n')[0]
            while filename:
                '''
                Inverted_table[word] = [files_id]
                A new table for every $checkpoint files
                '''
                inverted_table = dict()
                for word in words_list:
                    inverted_table[word] = []
                '''
                tf_table = [[tf-vector of file 1] .. [tf-vector of file n]]
                A new table for every $checkpoint files
                '''
                tf_table = list()
                while file_operation_ctr < self.checkpoint and filename:  # get to checkpoint
                    if file_ctr < checkpoint:
                        file_ctr += 1
                        filename = f.readline().split('\n')[0]
                        continue

                    file_path = os.path.join(self.dataset_path, filename)

                    words_tf = get_tf(file_path, self.error_log,
                                      file_ctr, cache, words_list)
                    # get tf vector of a word
                    words_tf_sorted = [words_tf[word]
                                       for word in self.words_list_sorted]
                    tf_table.append(words_tf_sorted)

                    for key in words_tf.keys():
                        if words_tf[key] > 0:  # which means the word occur in the file
                            inverted_table[key].append(file_ctr)

                    file_operation_ctr += 1
                    file_ctr += 1

                    if file_ctr % 1000 == 0:
                        t_end = time.process_time()
                        print(
                            file_ctr, " files have been visited, time cost:", t_end-t_start)
                        print("Cache length:", len(cache))
                        t_start = t_end

                    filename = f.readline().split('\n')[0]

                    # if file_ctr > 1:
                    #     return

                inverted_table_path = self.inverted_table + \
                    str(file_ctr) + ".csv"
                with open(inverted_table_path, 'w', newline='') as df:
                    f_csv = csv.writer(df)
                    for key in inverted_table.keys():
                        f_csv.writerow([key]+inverted_table[key])
                print(inverted_table_path, "has been saved.")

                '''
                To use less space, for every file, the tf vector is saved in this way:
                    if the i th component of the vector is 0, skip
                    else append i and vector[i] to the row in sequence.
                '''
                tf_table_path = self.tf_table + str(file_ctr) + ".csv"
                with open(tf_table_path, 'w', newline='') as df:
                    f_csv = csv.writer(df)
                    for i in range(len(tf_table)):
                        temp = list()
                        for j in range(len(tf_table[i])):
                            if tf_table[i][j] > 0:
                                temp.append(j)
                                temp.append(tf_table[i][j])
                        f_csv.writerow(temp)
                        # f_csv.writerow(tf_table[i])

                print(tf_table_path, "has been saved.")

                file_operation_ctr = 0

        print("Complete.")
        print("Total files:", file_ctr)

    def words_counting(self):
        cache = dict()  # Used to speed up tokenize process
        words = dict()  # words[word] = occur-time
        t_start = time.process_time()
        with open(self.filename_path, 'r') as f:
            file_ctr = 0
            filename = f.readline().split('\n')[0]
            while filename:
                file_ctr += 1
                file_path = os.path.join(self.dataset_path, filename)
                # words_in_file[word] = occur-time in file
                words_in_file = tokenize_file(
                    file_path, self.error_log, file_ctr, cache)
                for word in words_in_file:
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1

                if file_ctr % 1000 == 0:
                    t_end = time.process_time()
                    print(
                        file_ctr, " files have been visited, time cost:", t_end-t_start)
                    print("Cache length:", len(cache))
                    t_start = t_end

                filename = f.readline().split('\n')[0]

        result = sorted(
            words.items(), key=lambda item: item[1], reverse=True)[:1000]

        with open(self.words_list, 'w', newline='') as df:
            f_csv = csv.writer(df)
            for i in range(len(result)):
                f_csv.writerow(list(result[i]))
        print(self.words_list, "has been saved.")

        print("Complete.")
        print("Total files:", file_ctr)
        print("Total words:", len(words))

    '''
    Walk dataset_path, save the order in filename.csv
    Used in all following process
    '''

    def save_filename(self):
        with open(self.filename_path, 'w', newline='') as f:
            counter = 0
            f_csv = csv.writer(f)
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    counter += 1
                    if counter % 10000 == 0:
                        print(counter, " files have been visited")
                    f_csv.writerow(
                        [os.path.join(root.split(self.dataset_path)[-1], file)])

    '''
    Get the filename of all inverted_table's checkpoints
    '''

    def get_inverted_table_list(self):
        inverted_table_path = os.path.split(self.inverted_table)[0]
        outputs = os.listdir(inverted_table_path)
        inverted_tables = [output for output in outputs if re.match(
            os.path.split(self.inverted_table)[1]+"[0-9]+.*", output)]
        inverted_tables_num = [int(inverted_table.split(os.path.split(self.inverted_table)[
                                   1])[-1].split(".csv")[0]) for inverted_table in inverted_tables]
        inverted_tables_num.sort()
        inverted_table_sorted = list()
        for num in inverted_tables_num:
            for inverted_table in inverted_tables:
                if inverted_table.split(os.path.split(self.inverted_table)[1])[-1].split(".csv")[0] == str(num):
                    inverted_table_sorted.append(inverted_table)
                    break
        return inverted_table_sorted

    '''
    Get the filename of all tf_table's checkpoints
    '''

    def get_tf_table_list(self):
        tf_table_path = os.path.split(self.tf_table)[0]
        outputs = os.listdir(tf_table_path)
        tf_tables = [output for output in outputs if re.match(
            os.path.split(self.tf_table)[1]+"[0-9]+.*", output)]
        tf_tables_num = [int(tf_table.split(os.path.split(self.tf_table)[
                             1])[-1].split(".csv")[0]) for tf_table in tf_tables]
        tf_tables_num.sort()
        tf_table_sorted = list()
        for num in tf_tables_num:
            for tf_table in tf_tables:
                if tf_table.split(os.path.split(self.tf_table)[1])[-1].split(".csv")[0] == str(num):
                    tf_table_sorted.append(tf_table)
                    break
        return tf_table_sorted

    '''
    Merge all fragments of inverted_table
    '''

    def merge_inverted_table(self):
        inverted_tables = self.get_inverted_table_list()
        inverted_table_merged = dict()
        for inverted_table in inverted_tables:
            print("Merging " + inverted_table + "...")
            path = os.path.join(os.path.split(
                self.inverted_table)[0], inverted_table)
            with open(path, 'r') as f:
                table_line = f.readline().split('\n')[0]
                while table_line:
                    word = table_line.split(",")[0]
                    files_id = [int(id_str)
                                for id_str in table_line.split(",")[1:]]
                    if word in inverted_table_merged:
                        inverted_table_merged[word] += files_id
                    else:
                        inverted_table_merged[word] = files_id
                    table_line = f.readline().split('\n')[0]
        with open(self.inverted_table + '.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            for key in inverted_table_merged.keys():
                f_csv.writerow([key] + inverted_table_merged[key])

        print("Merging completed.")

    '''
    Read inverted_table, for every word, get number of files in which word occurs in.
    Use words_list_sorted to arrange it.
    This will be used in calculating tfidf-vector
    '''

    def get_word_occurence(self):
        word_occurence_dict = dict()
        with open(self.inverted_table + '.csv', 'r', newline='') as f:
            table_line = f.readline().split('\n')[0]
            while table_line:
                word = table_line.split(",")[0]
                files_id = [table_line.split(",")[1:]]
                word_occurence_dict[word] = len(files_id)
                table_line = f.readline().split('\n')[0]
        word_occurence = [word_occurence_dict[word]
                          for word in self.words_list_sorted]
        return word_occurence

    '''
    To use less space, for every file, the tf vector is saved in this way:
        if the i th component of the vector is 0, skip
        else append i and vector[i] to the row in sequence.

    Tfidf vector is also saved in this way.
    Need to calculate tfidf vector using tf vector and idf vector in this function

    Tfidf vector is not normalized in thid table!!!
    '''

    def get_tfidf_table(self):
        tf_tables = self.get_tf_table_list()

        with open(self.tfidf_table_path, 'w', newline='') as df:
            f_csv = csv.writer(df)
            for tf_table in tf_tables:
                print("Merging " + tf_table + "...")
                path = os.path.join(os.path.split(self.tf_table)[0], tf_table)
                with open(path, 'r') as f:
                    table_line = f.readline()
                    while table_line:
                        tfidf_table = list()
                        data = table_line.split("\n")[0].split(",")
                        for i in range(len(data)//2):
                            tfidf_table.append(data[2*i])
                            tfidf = float('%.03f' % (float(
                                data[2*i+1]) * math.log(self.file_num / self.word_occurence[int(data[2*i])], 10)))
                            tfidf_table.append(tfidf)
                        f_csv.writerow(tfidf_table)
                        table_line = f.readline()

            print("Merging completed.")

    '''
    Return the number of all files
    '''

    def get_file_num(self):
        with open(self.filename_path, 'r') as f:
            length = len(f.readlines())
        return length

    '''
    For a list of words, get its tfidf_vector
    The result is not normalized yet!
    '''

    def get_tfidf_vector(self, searching_words):
        limit = set(self.words_list_sorted)
        words_dict = dict()
        words_tf = dict.fromkeys(limit, 0)
        for i in range(len(self.words_list_sorted)):
            words_dict[self.words_list_sorted[i]] = i

        tfidf_vector = []
        for searching_word in searching_words:
            searching_word = PorterStemmer().stem(searching_word.lower())
            if searching_word in limit:
                if searching_word in words_tf:
                    words_tf[searching_word] += 1
                else:
                    words_tf[searching_word] = 1

        for word in words_tf.keys():
            if words_tf[word]:
                words_tf[word] = (1 + math.log(words_tf[word], 10))*math.log(
                    self.file_num / self.word_occurence[int(words_dict[word])], 10)

        for word in self.words_list_sorted:
            tfidf_vector.append(words_tf[word])
        return np.array(tfidf_vector)

    
def main():
    os.chdir(conf["WORKPATH"])
    dataloader = DataLoader(conf)
    dataloader.init()


if __name__ == '__main__':
    main()
