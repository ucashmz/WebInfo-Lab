import numpy as np
from dataLoader import DataLoader
from config import conf
import os

def vector_normalize(vector):
    len = np.sqrt(np.sum(np.square(vector)))
    if len == 0:
        return vector
    else:
        return vector/len

class TfidfSearch:
    def __init__(self, Dataset):
        self.dataset = Dataset
        self.dataset.init()
    '''
    Return first 10 files most related to searching words
    '''

    def _tfidf_search(self, searching_words):
        tfidf_vector = self.dataset.get_tfidf_vector(searching_words)
        # print(len(tfidf_vector))

        result = []
        with open(self.dataset.tfidf_table_path, 'r') as f:
            table_line = f.readline()
            ctr = 0
            while table_line:
                # data: [..., i, i.tfidf_value, ...]
                data = table_line.split("\n")[0].split(",")
                file_tfidf_vector = np.zeros(len(self.dataset.words_list_sorted))

                for i in range(len(data)//2):
                    file_tfidf_vector[int(data[i*2])] = float(data[2*i+1])

                dist = np.sqrt(np.sum(np.square(vector_normalize(
                    tfidf_vector) - vector_normalize(file_tfidf_vector))))

                result.append((ctr, dist))
                ctr += 1
                if ctr % 20000 == 0:
                    print(ctr, "files have been visited")
                table_line = f.readline()
                # print(result)
        print()
        result = [file[0] for file in sorted(result, key=lambda x:x[1])[:10]]
        return result

    def search(self):
        filename = list()
        with open(self.dataset.filename_path, 'r') as f:
            filename = f.readlines()

        while True:
            searching = input("(quit by input \'EXIT\')Search for:")
            if searching == 'EXIT':
                break
            else:
                separate_search_word = searching.split()
                files_id = self._tfidf_search(separate_search_word)
                print("Most related files:")
                for file_id in files_id:
                    print("\t", filename[file_id])

def main():
    os.chdir(conf["WORKPATH"])
    dataLoader = DataLoader(conf)
    tfidfSearch = TfidfSearch(dataLoader)
    tfidfSearch.search()


if __name__ == '__main__':
    main()