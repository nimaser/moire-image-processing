import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

plt.xkcd()

plt.xticks([])
plt.yticks([])
plt.xlim(0, 17)
plt.ylim(0, 17)

plt.xlabel("amount of rain")
plt.ylabel("amount of fun")
plt.title("rain vs fun while biking home")

x = np.linspace(0, 15, 16)
y = -0.15 * (x-15) * (x+5)
point = 8, 13.65

plt.plot(x, y, color="blue")
plt.plot(*point, marker="o", linestyle=" ", color="orange", label='"uh oh"')

plt.legend()
plt.show()