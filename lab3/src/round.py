def rounding(data):
    if data > 4.7:
        return 5
    elif data > 3.5:
        return 4
    elif data > 2.5:
        return 3
    elif data > 1.6:
        return 2
    elif data > 0.3:
        return 1
    else:
        return 0


with open("../output/out.out", 'r') as source, open("../output/out_rounding.out", 'w') as result:
    # with open("..\\output\\out_2.22.txt", 'r') as source, open("..\\output\\out_2.22_rounding2.txt", 'w') as result:
    count = 0
    while True:
        data = source.readline()
        if data:
            count += 1
            data = float(data)
            #print(count, data)
            data = rounding(data)
            # print(data)
            result.write(str(data)+"\n")
        else:
            break
        if count % 10000 == 0:
            print(count//10000)

print(count)
