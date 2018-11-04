import noise
import matplotlib.pyplot as plt

test = []
xoff = -8
from numpy import interp
size=10000

for i in range(size):
    n = noise.pnoise1(xoff, 6, 0.1, 3, 1024)
    n = interp(n, [-1, 1], [-0.13, 0.13])
    test.append(n)
    xoff += 0.0005

plt.plot(range(size), test)
plt.show()
