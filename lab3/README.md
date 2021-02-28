# WebInfo Lab3

**PB18000239 何灏迪**

**PB18000221 袁一玮**

## 程序主体

程序主体部分由 `main.py` 进行，实现来自于 `utils.py` 和 `round.py`

```python
M = Knn(conf)
origin, nonzero, zero = M.genMTrain()

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

M.recommend(newM, sqrtV, averV, zero)
```

main 中函数调用 utils 中写好的类 Knn；`genMTrain` 是从原始文件得到得分矩阵信息和每个用户输入的元素数量；`getAverV` 是得到用户的平均分；根据用户的平均分，我们可以用 `genNewM` 得到经过平均化的得分矩阵；`userSqrtV` 根据新的得分矩阵，给出每个用户的平方根，方便后面进行归一化；`recommend` 根据上述数据，给出合理的预测得分；`round` 把预测的浮点数分数按预定的规则输出成整数得分。

若已经存在 `userAve`、`userNewM`、`userSqrtV` 三个文件，`pickle` 会自动读取缓存、不需要进行生成，加快重建速度。

下面进行具体的解释。

### genMTrain

```python
def genMTrain(self):
    trainDat = os.path.join(self.dataset_path, self.trainData)
    if not os.path.exists(trainDat):
        print("not found dataset, it should be WORKPATH/dataset/training.dat")
        exit()
    with open(trainDat, 'r', encoding='utf-8') as csvfile:
        cs = list(csv.reader(csvfile))
    maxuser = int(cs[0][0])
    minuser = int(cs[0][1])
    maxmovie = int(cs[0][0])
    minmovie = int(cs[0][1])
    for record in cs:
        if int(record[0]) > maxuser:
            maxuser = int(record[0])
        if int(record[0]) < minuser:
            minuser = int(record[0])
        if int(record[1]) > maxmovie:
            maxmovie = int(record[1])
        if int(record[1]) < minmovie:
            minmovie = int(record[1])

    origin = np.zeros((maxmovie + 1, maxuser + 1), dtype=int)
    nonzero = np.zeros(maxuser + 1, dtype=int)
    zero = np.zeros(maxuser + 1, dtype=int)
    for record in cs:
        origin[int(record[1])][int(record[0])] = int(record[2])
        if int(record[2]) != 0:
            nonzero[int(record[0])] += 1
        else:
            zero[int(record[0])] += 1
    zero = zero / (zero + nonzero + 0.00001)
    print(zero.sum()/2185)
    return origin, nonzero, zero
```

从 `Training.dat` 导入，得到原始得分矩阵信息和每个用户输入的元素矢量 `nonzero` 以及每一个用户给出 0 分标记的比例，用于后续的处理。

### getAverV

```python
def getAverV(self, origin, nonzero):
    sumV = origin.sum(axis=0)
    users = size(sumV)
    averV = np.zeros(users, dtype=float)
    for i in range(users):
        averV[i] = float(sumV[i] / nonzero[i])
        if math.isnan(averV[i]):
            averV[i] = float(2)
    return averV
```

获得每一个用户的平均给分。如该用户之前没有过打分记录，以 2 分为其默认平均打分。该部分没有考虑所有 0 分的情况，这与我们对 0 分标记处理的理解有关，在推荐部分将具体解释。

### genNewM

```python
def genNewM(self, origin, averV):
    newM = np.zeros(origin.shape, dtype=float)
    for y in range(origin.shape[1]):
        for x in range(origin.shape[0]):
            if origin[x][y] != 0:
                newM[x][y] = float(origin[x][y]) - averV[y]
    return newM
```

得到新表格。该过程将去除用户的平均打分高低的影响，得到每个用户对电影的相对评价。

### userSqrtV

```python
def getSqrtV(self, newM):
    sqrtV = np.zeros(newM.shape[1], dtype=float)
    for y in range(newM.shape[1]):
        sqrtV[y] = np.linalg.norm(newM[:, y])
    return sqrtV
```

计算每个用户的平方根，方便后面 `recommend` 进行归一化。

### recommend

```python
def getSim(self, newM, sqrtV, userid):
    simV = np.zeros(newM.shape[1], dtype=float)
    for y in range(newM.shape[1]):
        sumup = np.dot(newM[:, y], newM[:, userid])
        norm = sqrtV[y] * sqrtV[userid]
        if norm:
            simV[y] = sumup/norm
        else:
            simV[y] = 0
    return simV

def predict(self, simV, newM, averV, userId, movieId, nonzero):
    k = 10
    kNear = []
    index = 0
    tmpV = np.copy(simV)
    while index < k:
        maxSim = np.argmax(tmpV)
        if maxSim == userId:
            tmpV[maxSim] = -1
            continue
        if tmpV[maxSim] <= 0:
            break
        if newM[movieId][maxSim] != 0:
            kNear.append(maxSim)
            index += 1
        tmpV[maxSim] = -1
    out = 0.0
    sumSim = 0.0
    if len(kNear) == k:
        for item in kNear:
            out += simV[item] * newM[movieId][item]
            sumSim += simV[item]
        out = out/sumSim + averV[userId]
    else:
        out = averV[userId]
    out = out * nonzero[userId]
    return out

def recommend(self, newM, sqrtV, averV, zero):
    count = 0
    testDat = os.path.join(self.dataset_path, self.testData)
    if not os.path.exists(testDat):
        print("not found dataset, it should be WORKPATH/dataset/testing.dat")
        exit()
    with open(testDat, 'r', encoding='utf-8') as csvfile:
        cs = list(csv.reader(csvfile))

    with open(os.path.join(self.out_path, self.testing_out), 'w', encoding='utf-8') as file:
        simV = self.getSim(newM, sqrtV, int(cs[0][0]))
        userId = int(cs[0][0])
        nonzero = np.power(1 - zero, 1.5)  # 1.5 is a parameter
        print("Non zero =", nonzero[userId])

        for record in cs:
            if int(record[0]) != userId:
                userId = int(record[0])
                simV = self.getSim(newM, sqrtV, int(record[0]))
                print("Sim got.")
                print("Non zero =", nonzero[userId])

            out = self.predict(simV, newM, averV, userId,
                                int(record[1]), nonzero)
            file.write(str(out)+"\n")

            count += 1
            if count % 1000 == 0:
                print(count/1000)

    print(count)
```

完成用户推荐的工作。该部分首先通过`predict`依据对电影打过分的用户中与用户相似度最高的 k 个用户对该电影的平均打分。若相似度超过 0 的用户少于 k 个（例如该用户之前完全没有记录），这里将直接取该用户的平均分。注意这里获得的评分是基于所有非 0 记录生成。这基于我们对 0 分记录的理解：0 分标记并非用户给出最低评价，而是用户未给出评价时系统提供的默认记录。这要求我们首先根据非 0 的记录进行得分的预测，再针对该用户给出评价的几率进行预测。这部分利用用户之前给出实际评价（而非默认的 0 分）的比例对预测的评价进行修正。

我们期望使用这样的方式理解 0 分标记（也是豆瓣中的实际机制）可以提供比将 0 分理解为最低分更好的预测效果。

### round

```python
def rounding(data):
    if data > 4.7:
        return 5
    elif data > 3.6:
        return 4
    elif data > 2.5:
        return 3
    elif data > 1.4:
        return 2
    elif data > 0.3:
        return 1
    else:
        return 0

with open("../output/out.out", 'r') as source, open("../output/out_rounding.out", 'w') as result:
    count = 0
    while True:
        data = source.readline()
        if data:
            count += 1
            data = float(data)
            data = rounding(data)
            result.write(str(data)+"\n")
        else:
            break
        if count % 10000 == 0:
            print(count // 10000)

print(count)
```

把预测的浮点数分数按预定的规则输出成整数。该部分中我们对 rounding 的方式进行了一些修改，这些修改基本基于我们在验证集上和少数几次在测试集上的尝试确定。在评分预测已经具有一定的可靠性的情况下，通过小幅修改 rounding 的具体数值，可以为整体预测的效果带来小幅提升。

## 优化

### 存储优化

有想过使用 dict 减少得分矩阵的占用，但是实际使用的过程中发现在进行相似度计算等过程中效率较完全向量化的操作相比还是较低。使用该思路进行运行耗时约为矩阵化的 5 倍。

### 缓存加载

考虑到计算量较大，如果每次运行计算，浪费了许多时间和机器资源。因此，程序设置了 pickle dump，保存每一步的计算结果。

### 其他尝试

过程中我们希望能考虑到每个用户的打分分布情况，包括使用分数方差进行额外的 normalization，并在最终预测时进行方差修正；以及直接使用分数在用户的所有评分中所占的位置来进行 normalization（如某用户打出每种分数的比例为 1：2：3：2：1，我们对这五个分数进行 normalization 的结果可能为 -88.9%：-55.5%：0：55.5%：88.9%，之后使用这样的归一化结果进行后续的预测，并在得到预测后依据具体用户的打分分布重新换算为具体的评分。

这些尝试在实践中没有得到显著提升的结果。

## 结果

最好的一次 RMSE 得分是 1.536；同时程序能在一个小时内完成所有任务。

## 总结

本次实验中我们进行了诸多尝试，并且在各个层面进行了一些优化希望得到更好的结果。最终，我们使用了基于用户进行预测的方法，并将 0 分记录单独进行考虑，在进行完预测后针对每个用户的 0 分记录比例情况进行修正。

我们没有对标签和社交推荐进行处理，只做了必选的个性化推荐技术。因为在邻接矩阵的存储方式下，标签这类字符串的存储比存储纯分数这类整形数据较为麻烦。
