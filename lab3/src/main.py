from config import conf
from utils import Knn
import pickle
import os


def main():
    os.chdir(conf["WORKPATH"])
    M = Knn(conf)

    origin, nonzero = M.genMTrain()

    userAve = os.path.join(conf["out_path"], conf["userAve"])
    if os.path.exists(userAve) and (os.path.getsize(userAve) > 0):
        with open(userAve, "rb") as file:
            averV = pickle.loads(file.read())
            print("load from", userAve)
    else:
        averV = M.getAverV(origin, nonzero)
        str = pickle.dumps(averV)
        with open(userAve, "wb") as file:
            file.write(str)
            print("save to", userAve)

    userNewM = os.path.join(conf["out_path"], conf["userNewM"])
    if os.path.exists(userNewM) and (os.path.getsize(userNewM) > 0):
        with open(userNewM, "rb") as file:
            newM = pickle.loads(file.read())
            print("load from", userNewM)
    else:
        newM = M.genNewM(origin, averV)
        str = pickle.dumps(newM)
        with open(userNewM, "wb") as file:
            file.write(str)
            print("save to", userNewM)
    
    userSqrtV = os.path.join(conf["out_path"], conf["userSqrtV"])
    if os.path.exists(userSqrtV) and (os.path.getsize(userSqrtV) > 0):
        with open(userSqrtV, "rb") as file:
            sqrtV = pickle.loads(file.read())
            print("load from", userSqrtV)
    else:
        sqrtV = M.getSqrtV(newM)
        str = pickle.dumps(sqrtV)
        with open(userSqrtV, "wb") as file:
            file.write(str)
            print("save to", userSqrtV)
    
    M.recommend(newM, sqrtV, averV)


if __name__ == '__main__':
    main()
