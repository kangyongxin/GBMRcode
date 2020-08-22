# 训练过程绘制
import matplotlib.pyplot as plt 
import numpy as np
# from gym import envs
# print(envs.registry.all())


# data = np.load("./results/e9r300b256dqn_mspacma.npy")
# data = np.load("./results/e9r300b256m5000dqn_mspacma.npy")
data = np.load("./results/e95r1000b256m5000dqn_mspacma.npy")
#data = np.load("dqn_mspacma.npy")
averagedata = []
th =200
for i in range(len(data)):
    if i<th:
        averagedata.append(sum(data[0:i])/(i+1))
    else:
        averagedata.append(sum(data[i-th+1:i]/th)) 
plt.plot(averagedata)
# plt.savefig("./results/e9r300b256dqn_mspacma.png")
# plt.savefig("./results/e9r300b256m5000dqn_mspacma.png")
plt.savefig("./results/e95r1000b256m5000dqn_mspacma.png")
#plt.savefig("dqn_mspacma.png")

# e9r300b256m5000

