import matplotlib.pyplot as plt 
import numpy as np 

data = np.loadtxt("timings.txt")

plt.plot(data[:,0], 1.0/data[:,1])
plt.savefig("timings.png", dpi=300)