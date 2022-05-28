import matplotlib.pyplot as plt
import numpy as np

f_loss = open("loss.txt", 'r')
f_val = open("val.txt", 'r')

X = X = np.arange(100)

l1 = []
l2 = []
l3 = []

v1 = []
v2 = []
v3 = []

for line in f_loss:
    line = line.split()
    l1.append(float(line[0]))
    l2.append(float(line[1]))
    l3.append(float(line[2]))

for line in f_val:
    line = line.split()
    v1.append(float(line[0]))
    v2.append(float(line[1]))
    v3.append(float(line[2]))


plt.plot(X, l1, color='r', label='stage1 loss')
plt.plot(X, l2, color='g', label='stage2 loss')
plt.plot(X, l3, color='b', label='stage3 loss')

plt.plot(X, v1, color='c', label='stage1 error')
plt.plot(X, v2, color='m', label='stage2 error')
plt.plot(X, v3, color='y', label='stage3 error')



plt.xlabel("Epochs")
plt.ylabel("")
plt.title("Training Loss and Validation Error")
plt.legend()
plt.savefig('loss-error.png')
