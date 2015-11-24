import scipy as sp
import time

n = 10**7
a = sp.random.binomial(1,0.1,n)
print a
t = time.time()
b = a.nonzero()
d = time.time()
print d - t
print b
i = 0
t1 = time.time()
while(i<n)and(a[i]!=0):
    i+=1
t2 = time.time()
print a[i]
