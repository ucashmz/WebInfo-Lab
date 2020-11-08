import csv, os, re, linecache, time
from config import conf
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# TODO error handling mode
# be used to try files in error_log again after error fixed

# TODO search


def read_file(file_path, error_log, counter):
    with open(file_path,'r',encoding='utf-8') as f:
        try:
            content = f.read()
        except:
            with open(error_log, 'a')as log:
                log.write(str(counter) + ",problems in " + file_path + "\n")
            print("problems in",file_path)
            return []
        paragraphs = content.split('\n\n')[1:] # discard header of the email
    return paragraphs

def tokenize_paragraph(paragraph, punctuation, stops):
    sentences = sent_tokenize(paragraph.lower())
    cut_word_sents = [word_tokenize(sentence) for sentence in sentences] # Maybe this could be done without nltk
    cut_word = list()
    for cut_word_sent in cut_word_sents:
        cut_word += cut_word_sent

    for word in cut_word[:]:
        if "." in word or "/" in word or '@' in word or '_' in word or '=' in word:
            cut_word.remove(word)
            cut_word += re.split(r'[./@_=]', word)
    cut_word = list(filter(None,cut_word))
    # print(cut_word)

    # This part could be simplified
    word_without_punc = [word for word in cut_word if word not in punctuation and not re.match('[*=_\'-/].*',word)] # to filter some words like '====='
    for i in range(len(word_without_punc)):
        while word_without_punc[i][-1] in ['=','-','*','/','_','\'']:
            # print(word_without_punc[i])
            word_without_punc[i] = word_without_punc[i][:-1]
            if not word_without_punc[i]:
                break

    word_without_stops = [word for word in word_without_punc if word not in stops]
    word_without_num = [word for word in word_without_stops if not re.match('.*[0-9].*',word)]
    word_stem = [PorterStemmer().stem(word) for word in word_without_num]
    return word_stem

def tokenize_file(file_path, error_log, counter):
    paragraphs = read_file(file_path, error_log, counter)
    punctuation = set([',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','...','-','--','=','..'])
    stops = set(stopwords.words("english")+['am','pm','ect','cc','ps','www','com']) # some stopwords added
    # print(stops)
    words = list()
    for paragraph in paragraphs:
        result = tokenize_paragraph(paragraph, punctuation, stops)
        words += result
    # print(words)
    return set(words)

class BoolSearch:
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.filename_path = config['filename_path']
        self.inverted_table = config['inverted_table']
        self.checkpoint = config['checkpoint']
        self.error_log = config['error_log']
    
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
            if os.path.exists(self.inverted_table + '.csv'):
                print("Found filename and inverted table. Start searching..")
            elif os.path.exists(self.inverted_table + str(self.get_file_num()) + '.csv'):
                print("All files visited. Incomplete inverted tables need to be merged.")
                self.merge_inverted_table()
            else:
                self.get_inverted_table()
                self.merge_inverted_table()
        else:
            if os.path.exists(self.inverted_table + '.csv') or self.get_inverted_table_list():
                print("ERROR: Without filename.csv while inverted table found. You may need to delete all inverted table and start again, otherwise the result may be wrong.")
                return
            else:
                self.save_filename()
                self.get_inverted_table()
                self.merge_inverted_table()

        self.search()
    
    def get_inverted_table(self):
        checkpoint = self.load_checkpoint()
        t_start = time.process_time()
        with open(self.filename_path, 'r') as f:
            file_ctr = 0
            file_operation_ctr = 0
            filename = f.readline().split('\n')[0]
            while filename:
                words = dict() # key=word value=index, use dict to reduce time complexity of finding word
                word_list = list() # [word1,word2,...] use word_list to get word by index
                inverted_table = list() # inverted_table[i] = [file_id for file if word_list[i] appears in file]
                word_ctr = 0
                while file_operation_ctr < self.checkpoint and filename: # save checkpoint 
                    if file_ctr < checkpoint:
                        file_ctr += 1
                        filename = f.readline().split('\n')[0]
                        continue

                    file_path = os.path.join(self.dataset_path, filename)
                    # print(file_path)
                    for word in tokenize_file(file_path, self.error_log, file_ctr):
                        if word in words:
                            inverted_table[words[word]].append(file_ctr)
                        else:
                            words[word] = word_ctr
                            word_list.append(word)
                            inverted_table.append([file_ctr])
                            word_ctr += 1

                    if file_ctr % 500 == 0:
                        t_end = time.process_time()
                        print(file_ctr," files have been visited, time cost:", t_end-t_start)
                        t_start = t_end
                        

                    file_operation_ctr += 1
                    file_ctr += 1
                    filename = f.readline().split('\n')[0]
                    
                    # if file_ctr > 1:
                    #     return

                inverted_table_path = self.inverted_table + str(file_ctr) + ".csv"
                with open(inverted_table_path, 'w', newline='') as df:
                    f_csv = csv.writer(df)
                    for i in range(len(word_list)):
                        f_csv.writerow([word_list[i]] + inverted_table[i])
                print(inverted_table_path, "has been saved.")
                file_operation_ctr = 0 

                # if file_ctr > 150000:
                #     break

        print("Complete.")
        print("Total files:", file_ctr)
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

    def get_inverted_table_list(self):
        inverted_table_path = os.path.split(self.inverted_table)[0]
        outputs = os.listdir(inverted_table_path)
        inverted_tables = [output for output in outputs if re.match(os.path.split(self.inverted_table)[1]+"[0-9]+.*", output)]
        inverted_tables_num = [int(inverted_table.split(os.path.split(self.inverted_table)[1])[-1].split(".csv")[0]) for inverted_table in inverted_tables]
        inverted_tables_num.sort()
        inverted_table_sorted = list()
        for num in inverted_tables_num:
            for inverted_table in inverted_tables:
                if inverted_table.split(os.path.split(self.inverted_table)[1])[-1].split(".csv")[0] == str(num):
                    inverted_table_sorted.append(inverted_table)
                    break
        return inverted_table_sorted

    def merge_inverted_table(self):
        inverted_tables = self.get_inverted_table_list()
        inverted_table_merged = list()
        word_ctr = 0
        words = dict()
        word_list = list()
        for inverted_table in inverted_tables:
            print("Merging " + inverted_table + "...")
            path = os.path.join(os.path.split(self.inverted_table)[0], inverted_table)
            with open(path, 'r') as f:
                table_line = f.readline().split('\n')[0]
                while table_line:
                    word = table_line.split(",")[0]
                    files_id = [int(id_str) for id_str in table_line.split(",")[1:]]
                    if word in words:
                        inverted_table_merged[words[word]] += files_id
                    else:
                        words[word] = word_ctr
                        word_ctr += 1
                        word_list.append(word)
                        inverted_table_merged.append(files_id)
                    table_line = f.readline().split('\n')[0]
        with open(self.inverted_table + '.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            for i in range(len(word_list)):
                f_csv.writerow([word_list[i]] + inverted_table_merged[i])
        print("Merged completed.")

    def get_file_num(self):
        with open(self.filename_path,'r') as f:
            length = len(f.readlines())
        return length

    def search(self):
        filename = list()
        
        with open(self.filename_path, 'r') as f:
            f_csv = csv.reader(f)
            for line in f_csv:
                filename.append(line[0])
        
        while True:
            bitmap = list()
            searching = input("(quit by input \'EXIT\')Search for:")
            if searching == 'EXIT':
                break
            searching_stem = PorterStemmer().stem(searching.lower())
            separate_search_stem = searching_stem.split()
            # just could deal ( a and b ) or c
            Found = False
            for item in separate_search_stem:
                if item == '(':
                    pass # stack?
                elif item == ')':
                    pass
                elif item == 'and':
                    pass
                elif item == 'or':
                    pass
                elif item == 'not':
                    pass
                else:
                    with open(self.inverted_table + '.csv', 'r') as f:
                        table_line = f.readline().split('\n')[0]
                        while table_line:
                            word = table_line.split(",")[0]
                            # if searching_stem == word:
                            if item == word:
                                # Found = True
                                tmp_index = 0
                                files_id = [int(id_str) for id_str in table_line.split(",")[1:]]
                                for file_id in files_id:
                                    tmp_index += 0b1 << (file_id - 1)
                                bitmap.append(tmp_index)
                                break

                            table_line = f.readline().split('\n')[0]

                        '''
                        if not Found:
                            print("Cannot found " + searching + " in any file.")
                        else:
                            files = [filename[file_id] for file_id in files_id]
                            print("Found in", len(files), "files")
                            if len(files) > 20:
                                print("Too much to be shown on screen..")
                            else:
                                for file in files:
                                    print("\t", file)
                        '''

def main():
    os.chdir(conf["WORKPATH"])
    print("dataset:",conf["dataset_path"])
    print("filename_path:",conf["filename_path"])
    bs = BoolSearch(conf)
    bs.run()
    
if __name__ == '__main__':
    main()