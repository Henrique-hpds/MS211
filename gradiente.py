import numpy as np
from matplotlib import pyplot as plt

erro = 10e-4

A = np.array([[93, 24], [24, 107]])
b = np.array([42, 31])
F = lambda x, y: (A[0][0] * x**2 + A[1][1] * y**2)/2 + A[0][1] * x * y - b[0] * x - b[1] * y

dom = np.linspace(0, 0.5, 1000)
X, Y = np.meshgrid(dom, dom)
Z = F(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(X, Y)')

ax.set_xlim(0.3, 0.5)
ax.set_ylim(0.1, 0.4)
ax.set_zlim(-12, -11)

ponto = [0, 0]

residuo = lambda x: b - np.dot(A, x)

while residuo(ponto).any() > erro:
    alpha_teorico = np.dot(residuo(ponto), residuo(ponto)) / np.dot(np.dot(A, residuo(ponto)), residuo(ponto))
    gradiente = np.dot(A, ponto) - b
    ponto = ponto - alpha_teorico * gradiente
    ax.scatter(ponto[0], ponto[1], F(ponto[0], ponto[1]), color='r')


plt.show()