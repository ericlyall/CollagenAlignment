import numpy as np

a=[1,1]
b=[1,3]
c=np.add(a,b)
print(c)

d=[a[1],a[0]*-1]
print(d)

ang_mult= np.arange(180)
angles= [np.pi/180*x for x in ang_mult]

sub180= [180-y for y in ang_mult]
print(sub180)