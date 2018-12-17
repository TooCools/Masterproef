import noise
import matplotlib.pyplot as plt

test = []
xoff = -8
from numpy import interp
size=10000
#-8 voor val
#19 voor data

for i in range(size):
    n = noise.pnoise1(xoff, 6, 0.1, 3, 1024)
    n = interp(n, [-1, 1], [-0.02, 0.07])
    test.append(n)
    xoff += 0.0005

plt.plot(test)
plt.xlabel("Tijd")
plt.ylabel("Helling")
plt.show()
