import matplotlib.pyplot as plt 
import numpy as np
from gym import envs
print(envs.registry.all())


data = np.load("rewstep.npy")
averagedata = []
th =1000
for i in range(len(data)):
    if i<th:
        averagedata.append(sum(data[0:i])/(i+1))
    else:
        averagedata.append(sum(data[i-th+1:i]/th)) 
plt.plot(averagedata)
plt.savefig("rewstep.png")

