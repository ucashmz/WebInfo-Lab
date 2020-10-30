import csv, os, nltk
from config import conf

def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        content = f.read()
        paragraphs = content.split('\n\n')[1:]
    return(paragraphs)



class BoolSearch:
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.filename_path = config['filename_path']
    
    def run(self):
        if os.path.exists(self.filename_path):
            print("Load from", self.filename_path)
        else:
            self.save_filename()
        self.load_filename()
    
    def load_filename(self):
        with open(self.filename_path, 'r')as f:
            file_path = os.path.join(self.dataset_path,f.readlines()[0].split('\n')[0])
            read_file(file_path)

    def save_filename(self):
        with open(self.filename_path, 'w', newline='') as f:
            counter = 0
            f_csv = csv.writer(f)
            for root, _, files in os.walk(self.dataset_path):
                for file in files:
                    counter += 1
                    if counter%10000 == 0:
                        print(counter," files have been visited")
                    f_csv.writerow([os.path.join(root.split(self.dataset_path)[-1], file)])



def main():
    os.chdir(conf["WORKPATH"])
    print(conf["dataset_path"])
    print(conf["filename_path"])
    bs = BoolSearch(conf)
    bs.run()
    

if __name__ == '__main__':
    main()