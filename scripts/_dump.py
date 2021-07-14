import numpy as np

import pandas as pd


l = [22, 13, 45, 50, 98, 69, 43, 44, 1]
aa =  [x+1 if x >= 45 else x+5 for x in l]
print(aa)
#a.append(-float(b))

#print(a)
import re

string = 'A'

# Three digit number followed by space followed by two digit number
pattern = '(^[a-z|A-Z]{1})(\-?)(\d*$)'

# match variable contains a Match object.
match = re.search(pattern, string)

if match:
    print("all")
    print(match.group())

    print("Group 1")
    print(match.group(1))
    print("Group 2")

    print(match.group(2))

    if match.group((2)):
        print("yes")
    else:
        print("nsdofnjsdof")
    print("Group 3")

    print(match.group(3))
    print("Group end")
else:
  print("pattern not found")

# Output: 801 35


# Output: ['Twelve:', ' Eighty nine:', '.']


"""
n_num = 3
l = []
a = [[],[],[]]
aa = [[]for i in range(n_num)]


#a = [l]*n_num


print(a)
print(aa)
print("data a", np.shape(a))
print("data aa", np.shape(aa))


a[0].append([0.0, 0])
a[0].append([1.0, 0])
a[0].append([2.0, 0])

a[1].append([0.0, 1])
a[1].append([1.0, 1])
a[1].append([2.0, 1])

a[2].append([0.0, 2])
a[2].append([1.0, 2])
a[2].append([2.0, 2])


def _getme (v,d):
    print("-------")
    print( "voice ",v, "duration ",d)
    print(a[v][d])
    print("-------")



_getme(2, 2) # [1, 0]


voices = 3
nrow x voices x 2 
0.0 1   0.0 2   0.0 3
1.0 1   1.0 2   1.0 3
2.0 1   2.0 2   2.0 3


a[1].append([0.0, 1])
a[1].append([1.0, 1])
a[1].append([2.0, 1])

a[2].append([0.0, 2])
a[2].append([1.0, 2])
a[2].append([2.0, 2])

print(pd.DataFrame(a))

print(a)
print(np.shape(a))


# get me 2 value of voice 3


t = [[[0.0, 0], [0.1, 0], [0.2, 0]],
     [[0.0, 1], [0.1, 1], [0.2, 1]],
     [[0.0, 2], [0.1, 2], [0.2, 2]],
     [[0.0, 3], [0.1, 3], [0.2, 3]]]

t = [[[0.0, 0], [0.1, 0], [0.2, 0]],
     [[0.0, 1], [0.1, 1], [0.2, 1]],
     [[0.0, 2], [0.1, 2], [0.2, 2]],
     [[0.0, 3], [0.1, 3], [0.2, 3]]]

t = [[[1,1], [2,1], [3,1]], [2], [3]]

print("sdasd",np.shape(t))
#t = [[], [], []]
print(t)
print(t[0])
t[0].append([4,1])
print(t)
print(t[0])
print("sdasd",np.shape(t))

"""