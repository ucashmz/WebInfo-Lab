from nltk.stem import PorterStemmer
from dataLoader import DataLoader
from config import conf
import os
import linecache


def isOP(ch):
    if ch == "AND" or ch == "OR" or ch == "NOT":
        return True
    elif ch == "(" or ch == ")" or ch == "#":
        return True
    else:
        return False

def precede(a, b): # TODO add NOT
    if b == "AND":
        if a == "OR" or a == "#" or a == "(":
            return '<'
        elif a == "AND" or a == ")" or a == "NOT":
            return '>'
    elif b == "OR":
        if a == "#" or a == "(":
            return '<'
        elif a == ")" or a == "OR" or a == "AND" or a == "NOT":
            return '>'
    elif b == "NOT":
        if a == "AND" or a == "OR" or a == "(" or a == "#":
            return '<'
    elif b == "(":
        if a == "AND" or a == "OR" or a == "NOT" or a == "(" or a == "#":
            return '<'
    elif b == ")":
        if a == "AND" or a == "OR" or a == "NOT" or a == ")":
            return '>'
        elif a == "(":
            return '='
    elif b == "#":
        if a == "#":
            return '='
        else:
            return '>'
    print("Error input")
    sys.exit(main, 0)

def operate(a, theta, b):
    if theta == "AND":
        return op_and(a, b)
    elif theta == "OR":
        return op_or(a, b)
    print("Error input")
    sys.exit(main, 0)

def op_not(a: list, total_num):
    '''
    all_index = list(range(517402))
    for i in a:
        all_index.remove(int(i))
    return all_index
    '''
    index = list()
    i = 0
    size = len(a)
    for r in range(total_num):
        if i < size and str(r) == a[i]:
            i += 1
            continue
        index.append(str(r))
    return index

def op_and(a: list, b: list):
    seen = set()
    duplicated = set()

    for x in a+b:
        if x not in seen:
            seen.add(x)
        else:
            duplicated.add(x)
    return list(duplicated)

def op_or(a, b):
    return list(set(a + b))

class BoolSearch:
    def __init__(self, Dataset):
        self.dataset = Dataset
        self.dataset.init()
   
    '''
    Search a word.
    Words_dict is organized as: words_dict[word] = i, while i is the line of the word in inverted table
    Return a list of files_id in which the word occurs
    '''
    def _search(self, searching_word, words_dict):
        searching_stem = PorterStemmer().stem(searching_word.lower())
        if searching_stem in words_dict:
            result = linecache.getline(
                self.dataset.inverted_table + '.csv', words_dict[searching_stem]).split(',')[1:]
            linecache.clearcache()
        else:
            result = []
        return result

    def search(self):
        filename = list()
        words_dict = dict()
        with open(self.dataset.filename_path, 'r') as f:
            filename = f.readlines()
        files_num = len(filename)
        print("Loading inverted table..")

        '''
        load words_dict from inverted_table
        '''
        with open(self.dataset.inverted_table + '.csv', 'r') as f:
            table_line = f.readline().split('\n')[0]
            ctr = 1
            while table_line:
                word = table_line.split(",")[0]
                words_dict[word] = ctr
                ctr += 1
                table_line = f.readline().split('\n')[0]
        # print(words_dict)

        while True:
            print("Please input one space between all words(include parenthesis/brace and items).")
            searching = input("(quit by input \'EXIT\')Search for:")
            if searching == 'EXIT':
                break
            else:
                searching = searching.split(" ")
                searching.append("#")
                opnd = list()
                optr = list()
                optr.append("#")
                i = 0
                item = searching[0]
                search_size = len(searching)
                while (item != "#" or optr[-1] != "#") and i < search_size:
                    if not isOP(item):
                        input_list = self._search(item, words_dict)
                        opnd.append(input_list)
                        i += 1
                        item = searching[i]
                    else:
                        prior = precede(optr[-1], item)
                        if prior == "<":
                            optr.append(item)
                            i += 1
                            item = searching[i]
                        elif prior == ">":
                            theta = optr.pop()
                            if theta == "NOT":
                                b = opnd.pop()
                                opnd.append(op_not(b, files_num))
                            else:
                                b = opnd.pop()
                                a = opnd.pop()
                                opnd.append(operate(a, theta, b))
                        else:
                            optr.pop()
                            i += 1
                            item = searching[i]
                result = opnd[-1]

                if len(result) > 0:
                    print("Found in", len(result), "files. First found in", filename[int(result[0])])
                else:
                    print("Not found.")

def main():
    os.chdir(conf["WORKPATH"])
    dataLoader = DataLoader(conf)
    boolSearch = BoolSearch(dataLoader)
    boolSearch.search()


if __name__ == '__main__':
    main()