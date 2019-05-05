import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

'''
Read data from csv
'''
df = pd.read_csv('iris.csv')
df = df.drop(['Id'],axis=1)
target = df['Species']
rows = list(range(100,150))
# shape: (100, 5). (100 samples, 4 feature + 1 labels)
df = df.drop(df.index[rows])

df = df.drop(['SepalWidthCm','PetalWidthCm'], axis=1)
# shape: (100, 3). (100 samples, 2 feature + 1 labels)

Y = []
target = df['Species']
for val in target:
    if(val == 'Iris-setosa'):
        Y.append(-1)
    else:
        Y.append(1)
df = df.drop(['Species'],axis=1)
X = df.values.tolist()
## Shuffle and split the data into training and test set
X, Y = shuffle(X,Y)
'''
with open("in.txt", "w") as f:
    f.write("100\n")
    for i in range(100):
        line = str(X[i][0]) + " " + str(X[i][1]) + " "  + str(Y[i]) + "\n"
        f.write(line)
'''
x_train = []
y_train = []
x_test = []
y_test = []

# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2)


x_train = np.zeros((20, 2))
x_test = np.zeros((80, 2))
y_train = np.zeros(20)
y_test = np.zeros(80)
with open("in.txt", "r") as f:
    lineNum = -2
    for line in f:
        lineNum += 1
        if lineNum == -1:
            continue
        nums = line.strip().split(' ')
        if lineNum < 20:
            x_train[lineNum][0] = float(nums[0])
            x_train[lineNum][1] = float(nums[1])
            y_train[lineNum] = int(nums[2])
        else:
            idx = lineNum - 80
            x_test[idx][0] = float(nums[0])
            x_test[idx][1] = float(nums[1])
            y_test[idx] = int(nums[2])


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)



# print (x_train)

y_train = y_train.reshape(20,1)
y_test = y_test.reshape(80,1)



'''
Training process
'''

train_f1 = x_train[:,0]
train_f2 = x_train[:,1]

train_f1 = train_f1.reshape(20,1)
train_f2 = train_f2.reshape(20,1)

w1 = np.zeros((20,1))
w2 = np.zeros((20,1))

epochs = 1
alpha = 0.0001


while(epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    count = 0
    # My implementation
    
    update1 = - (2 * 1/epochs * w1)
    update2 = - (2 * 1/epochs * w2)
    for val in prod:
        if (val < 1):
            update1 += train_f1[count] * y_train[count]
            update2 += train_f2[count] * y_train[count]
        count += 1
    w1 = w1 + alpha * update1
    w2 = w2 + alpha * update2
    epochs += 1


'''
Testing Process
'''
## Clip the weights 
'''
index = list(range(80,20))

w1 = np.delete(w1,index)
w2 = np.delete(w2,index)
print (w1)
print (w2)

w1 = w1.reshape(80,1)
w2 = w2.reshape(80,1)
'''
## Extract the test data features 
test_f1 = x_test[:,0]
test_f2 = x_test[:,1]

test_f1 = test_f1.reshape(80,1)
test_f2 = test_f2.reshape(80,1)
## Predict
y_pred = w1[0] * test_f1 + w2[0] * test_f2
predictions = []
for val in y_pred:
    if(val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)

print(accuracy_score(y_test,predictions))