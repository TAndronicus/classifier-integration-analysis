import numpy as np
import matplotlib.pyplot as plt

a11, a12, a13, b11, b12, b13 = .25, .5, .75, 0, .5, -1
x1 = np.linspace(0, 10)
y11, y12, y13 = a11 * x1 + b11, a12 * x1 + b12, a13 * x1 + b13
ax = plt.subplot(1, 2, 1)
ax.plot(x1, y11, color = 'r')
ax.plot(x1, y12, color = 'g')
ax.plot(x1, y13, color = 'b')
plt.title('Po selekcji')
bx = plt.subplot(1, 2, 2)
bx.plot(np.linspace(2, 10), a11 * np.linspace(2, 10) + b11, color = 'xkcd:silver')
bx.plot(np.linspace(0, 6), a12 * np.linspace(0, 6) + b12, color = 'xkcd:silver')
bx.plot(np.linspace(0, 2), a13 * np.linspace(0, 2) + b13, color = 'xkcd:silver')
bx.plot(np.linspace(6, 10), a13 * np.linspace(6, 10) + b13, color = 'xkcd:silver')
bx.plot(np.linspace(0, 2), a11 * np.linspace(0, 2) + b11, color = 'xkcd:red')
bx.plot(np.linspace(2, 6), a13 * np.linspace(2, 6) + b13, color = 'xkcd:red')
bx.plot(np.linspace(6, 10), a12 * np.linspace(6, 10) + b12, color = 'xkcd:red')
plt.title('Po integracji')
plt.show()

a1, a2, a3, a4, b1, b2, b3, b4 = .25, .5, .75, 1, 0, 1.5, -1, -3
x = np.linspace(0, 10)
y1, y2, y3, y4 = a1 * x + b1, a2 * x + b2, a3 * x + b3, a4 * x + b4
ax = plt.subplot(1, 2, 1)
ax.plot(x, y1, color = 'xkcd:red')
ax.plot(x, y2, color = 'xkcd:green')
ax.plot(x, y3, color = 'xkcd:blue')
ax.plot(x, y4, color = 'xkcd:black')
plt.title('Po selekcji')
bx = plt.subplot(1, 2, 2)
bx.plot(x, y1, color = 'xkcd:silver')
bx.plot(x, y2, color = 'xkcd:silver')
bx.plot(x, y3, color = 'xkcd:silver')
bx.plot(x, y4, color = 'xkcd:silver')
bx.plot(np.linspace(0, 4), np.linspace(0, 4) * (a1 + a3) / 2 + (b1 + b3) / 2, color = 'xkcd:red')
bx.plot(np.linspace(4, 9), np.linspace(4, 9) * (a4 + a3) / 2 + (b4 + b3) / 2, color = 'xkcd:red')
bx.plot(np.linspace(9, 10), np.linspace(9, 10) * (a3 + a2) / 2 + (b3 + b2) / 2, color = 'xkcd:red')
plt.title('Po integracji')
plt.show()

a1, a2, b1, b2 = 1, 2, 1, -4
w1, w2 = 1, .3
x = np.linspace(0, 10)
y1 = a1 * x + b1
y2 = a2 * x + b2
ax = plt.subplot(1, 2, 1)
ax.plot(x, y1, color = 'g')
ax.plot(x, y2, color = 'b')
ax.legend(['ACC = 1', 'ACC = 0.3'])
plt.title('Po selekcji')
bx = plt.subplot(1, 2, 2)
bx.plot(x, y1, color = 'xkcd:silver')
bx.plot(x, y2, color = 'xkcd:silver')
bx.plot(x, (w1 * y1 + w2 * y2) / (w1 + w2), color = 'xkcd:red')
plt.title('Po integracji')
plt.show()