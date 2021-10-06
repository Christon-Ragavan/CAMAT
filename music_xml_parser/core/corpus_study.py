import numpy as np

a = np.array([[4,7], [1,2]])

print(a)
a.sort(axis=1)
print(a)

a.sort(axis=0)
print(a)
print(a)