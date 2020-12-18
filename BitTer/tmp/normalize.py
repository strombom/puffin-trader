
import numpy as np
from scipy.optimize import curve_fit


def func(x, a, b, c, d, e, f):
    #return a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5

    return a * x ** -b + c #+ d ** x

x_data = np.array([0.039, 0.033, 0.027, 0.022, 0.018, 0.015, 0.012, 0.01, 0.0082, 0.0068, 0.0065, 0.0047, 0.0039, 0.0033, 0.0027, 0.0022])
y_data = np.array([0.039, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, 0.85, 1, 1.15, 1.3, 1.5, 2, 2.25, 2.5, 2.8])

p_opt, p_cov = curve_fit(func, x_data, y_data)

print('coeff', p_opt)

y_out = func(x_data, *p_opt)

for lmnt in y_out:
    print(lmnt + 0.14)


#  0.1344193 * x ** -0.54606854 - 0.89771839
# a * x ** -b + c
