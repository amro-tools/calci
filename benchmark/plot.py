import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("timings.txt")

fig, ax = plt.subplots(1, 2)

ax[0].plot(
    data[:, 0],
    1.0 / data[:, 1],
    color="blue",
    label="Calci (cached neighbour list)",
    marker=".",
)
ax[0].plot(
    data[:, 0],
    1.0 / data[:, 2],
    color="tab:orange",
    label="Calci (rebuilt neighbour list)",
    marker=".",
)
ax[0].grid(True)
ax[0].axhline(1.0 / data[0, 3], linestyle="-", color="red", label="ASE")
ax[0].set_xlabel("Calci threads")
ax[0].set_ylabel("Energy & Force calculations per second")

ax[1].plot(
    data[:, 0],
    data[0, 3] / data[:, 1],
    marker=".",
    color="blue",
)
ax[1].plot(
    data[:, 0],
    data[0, 3] / data[:, 2],
    marker=".",
    color="tab:orange",
)
ax[1].set_xlabel("Calci threads")
ax[1].set_ylabel("Speedup")
ax[1].grid(True)
fig.legend(loc="upper center")
fig.tight_layout()
plt.savefig("timings.png", dpi=300)
