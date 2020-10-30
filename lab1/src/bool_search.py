import csv, os, re, linecache
from config import conf
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        try:
            content = f.read()
        except:
            print("problems in",file_path)
            return []
        paragraphs = content.split('\n\n')[1:] # discard header of the email
    return paragraphs

def tokenize_paragraph(paragraph, punctuation, stops):
    cut_word = word_tokenize(paragraph.lower())
    word_without_punc = [word for word in cut_word if word not in punctuation and word[0]!="'" and word[0]!="_"]
    word_without_stops = [word for word in word_without_punc if word not in stops]
    word_without_num = [word for word in word_without_stops if not re.match('.*[0-9].*',word)]
    word_stem = [PorterStemmer().stem(word) for word in word_without_num]
    return set(word_stem)

def tokenize_file(file_path):
    paragraphs = read_file(file_path)
    punctuation = set([',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','...','-','--'])
    stops = set(stopwords.words("english")+['am','pm','ect','cc','ps'])
    # print(stops)
    words = set()
    for paragraph in paragraphs:
        result = tokenize_paragraph(paragraph, punctuation, stops)
        words = words.union(result)
    # print(words)
    return words

class BoolSearch:
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.filename_path = config['filename_path']
        self.inverted_table = config['inverted_table']
        self.checkpoint = config['checkpoint']
    
    def load_checkpoint(self):
        i = 0
        while (os.path.exists(self.inverted_table + str(i + self.checkpoint) + ".csv")):
            i += self.checkpoint
        if i!=0:
            print("Load from checkpoint", i)
        else:
            print("Start without checkpoint.")
        return i

    def run(self):
        if os.path.exists(self.filename_path):
            print("Load from", self.filename_path)
        else:
            self.save_filename()
        self.get_inverted_table()
    
    def get_inverted_table(self):
        checkpoint = self.load_checkpoint()
        with open(self.filename_path, 'r')as f:
            file_ctr = 0
            file_operation_ctr = 0
            filename = f.readline().split('\n')[0]
            while filename:
                words = dict()
                word_list = list()
                inverted_table = list()
                word_ctr = 0
                while file_operation_ctr < self.checkpoint and filename:
                    if file_ctr < checkpoint:
                        file_ctr += 1
                        filename = f.readline().split('\n')[0]
                        continue

                    file_path = os.path.join(self.dataset_path, filename)
                    # print(file_path)
                    for word in tokenize_file(file_path):
                        if word in words:
                            inverted_table[words[word]].append(file_ctr)
                        else:
                            words[word] = word_ctr
                            word_list.append(word)
                            inverted_table.append([file_ctr])
                            word_ctr += 1

                    if file_ctr % 500 == 0:
                        print(file_ctr," files have been visited")

                    file_operation_ctr += 1
                    file_ctr += 1
                    filename = f.readline().split('\n')[0]

                inverted_table_path = self.inverted_table + str(file_ctr) + ".csv"
                with open(inverted_table_path, 'w', newline='') as df:
                    f_csv = csv.writer(df)
                    for i in range(len(word_list)):
                        f_csv.writerow([word_list[i]] + inverted_table[i])
                print(inverted_table_path, "has been saved.")
                file_operation_ctr = 0 

        print("Complete.")
        print("Total files:", file_ctr - 1)
        print("Total words:", word_ctr)


        

        


    def save_filename(self):
        with open(self.filename_path, 'w', newline='') as f:
            counter = 0
            f_csv = csv.writer(f)
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    counter += 1
                    if counter % 10000 == 0:
                        print(counter," files have been visited")
                    f_csv.writerow([os.path.join(root.split(self.dataset_path)[-1], file)])



def main():
    os.chdir(conf["WORKPATH"])
    print("dataset:",conf["dataset_path"])
    print("filename_path:",conf["filename_path"])
    bs = BoolSearch(conf)
    bs.run()
    

if __name__ == '__main__':
    main()