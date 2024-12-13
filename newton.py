import matplotlib.pyplot as plt
import numpy as np

def newton(x0, err, func, dfunc):

    xk = xk1 = x0
    iter = 0

    while abs(func(xk1)) > err:
        plt.scatter(xk1, func(xk1))
        print(xk1, func(xk1))
        xk1 = xk - func(xk)/dfunc(xk)
        xk = xk1
        iter += 1

    return xk1, iter

x0 = 1.0
err = 1.0e-6
func = lambda x: x**2 - 2
dfunc = lambda x: 2*x

print(newton(x0, err, func, dfunc))
x_axis = np.linspace(0, 2, 1000)
plt.plot(x_axis, func(x_axis))
plt.plot(x_axis, np.zeros(1000))
plt.show()