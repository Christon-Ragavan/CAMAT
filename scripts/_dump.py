# import matplotlib.pyplot as plt
# import numpy as np
# import math
# x = np.arange(0, math.pi*2, 0.05)
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
# y = np.sin(x)
# ax.plot(x, y)
# ax.set_xlabel('angle')
# ax.set_title('sine')
# ax.set_yticks([-1,0,1])
# ax.set_xticks([0,2,4,6])
# ax.set_xticklabels(['zero', None,'four','six'])
# plt.show()
#

import numpy as np
import matplotlib.pyplot as plt

ind = np.arange(3)
width = .2

x = list()
# x labels position: i = 1st bar, i+w/2 = category, i+w = 2nd bar
for i in ind:
    x.extend([i, i+width/2., i+width])
print(x)
# plot bars
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, [1, 3, 5], width, color='r', align = 'center')
rects2 = ax.bar(ind+width, [2, 4, 6], width, color='g', align = 'center')
# set ticks and labels
plt.xticks(x)
lab = ['A1','\n\nGeneral Info', 'A2', 'B1','\n\nTechnical', 'B2', 'C1','\n\nPsycological', 'C2']
l1 = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
l2 = np.arange(len(l1))
l2 = [str(i) for i in l2]
l22 = []
for i in range(0, len(l2)):
    print(i)
    li1 = l1[i]
    li2 = '\n\n'+l2[i]
    l22.append(li1)
    l22.append(li2)


print(l22)
print(len(l22))
print(len(lab))



ax.set_xticklabels(lab,ha='center')
# hide tick lines for x axis
ax.tick_params(axis='x', which='both',length=0)
# rotate labels with A
for label in ax.get_xmajorticklabels():
    if 'A' in label.get_text(): label.set_rotation(45)

plt.show()