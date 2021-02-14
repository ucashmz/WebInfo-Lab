from config import conf
from utils import Knn
import os


def main():
    os.chdir(conf["WORKPATH"])
    M = Knn(conf)
    origin, nonzero = M.genFromTrain()
    averV = M.getAver(origin, nonzero)
    print(averV)


if __name__ == '__main__':
    main()
