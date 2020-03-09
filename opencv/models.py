import math
import numpy as np

a = 5.  # in m/sec2
v = 30. # in km/h
d = 25. # in m
v = 30 * 1000 / (60.0 * 60.0)
tlc_cam = math.sqrt(2 * d / (3 * a))
print(tlc_cam)

coeff = [a/2, v, -d]
roots = (np.roots(coeff))
print(roots)
print(pow(roots[0], 2) * a/2 + roots[0] * v - d)
print(pow(roots[1], 2) * a/2 + roots[1] * v - d)

tlc_cvm = (d/(v))
print(tlc_cvm)
