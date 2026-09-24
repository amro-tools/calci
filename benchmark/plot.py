import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("timings.txt")

fig, ax = plt.subplots(1, 2)

ax[0].plot(data[:, 0], 1.0 / data[:, 1], label="Calci", marker=".")
ax[0].axhline(1.0 / data[0, 2], color="tab:orange", linestyle="--", label="ASE")
ax[0].set_xlabel("Calci threads")
ax[0].set_ylabel("Runs per second")
ax[0].grid(True)
ax[0].legend()


ax[1].plot(
    data[:, 0], data[0, 2] / data[:, 1], label="Calci", marker=".", color="black"
)
ax[1].set_xlabel("Calci threads")
ax[1].set_ylabel("Speedup")
ax[1].grid(True)
ax[1].legend()


plt.legend()
plt.tight_layout()
plt.savefig("timings.png", dpi=300)
