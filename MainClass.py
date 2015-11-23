import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

class Subway():
    T = None
    lamda = None
    n = None
    time_run = None
    p = None
    def __init__(self,T,lamda,n,time_run):
        self.T = T
        self.lamda = lamda
        self.n = n
        self.time_run = time_run
        self.p = float(self.lamda * self.T)/self.n

    def Calculate_Poisson_PMF(self):
        #time slot
        x = np.arange(1,self.time_run,1)

        #random with binomial parameter lambda
        arri = sp.random.binomial(1,self.p,[self.time_run,self.n])

        #history list
        hist_arrival = np.zeros(self.time_run)
        for i in range(self.time_run):
            summ = 0
            for j in range(self.n):
                summ+=arri[i][j]
            hist_arrival[summ]+=1

        return hist_arrival/self.time_run,poisson.pmf(x,self.lamda*self.T)


    def pdf_arrival(self):

        step = 0

        x = np.arange(1,self.time_run,1)
        hist_time = np.zeros(self.time_run)
        arri = sp.random.binomial(1,self.p,[self.time_run,self.n])
        while(step < self.time_run):
            time = 0
            while(time < n)and(arri[step][time]==0):
                time+=1
            if(time<n):
                hist_time[time] += 1
            step+=1
        print(sum(hist_time))
        return hist_time/self.time_run,sp.stats.expon.pdf(x,self.lamda)

    def cdf_arrival(self):
        step = 0
        #time slot
        x = np.arange(1,self.time_run,1)

        hist_time = np.zeros(self.time_run)


        arri = sp.random.binomial(1,self.p,[self.time_run,self.n])

        while(step < self.time_run):
            time = 0
            while(time < n)and(arri[step][time]==0):
                time+=1
            if(time<n):
                hist_time[time] += 1
            step+=1
        hist_time = hist_time/self.time_run
        return np.cumsum(hist_time),sp.stats.expon.cdf(x,self.lamda)


T = 10
lamda = 1
n = 10**2
times_run = 10**5

#Question A
plt.figure(1)
a = Subway(T,lamda,n,times_run)
cal,ppdf = a.Calculate_Poisson_PMF()
plt.plot(cal[0:51],'bo-',label='Calculated')
plt.plot(ppdf[0:51],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('pmf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')
#plt.show()

plt.figure(2)
cal2,expopdf = a.pdf_arrival()
cal2[0] = 1
plt.plot(cal2[0:50],'bo-',label='Calculated')
plt.plot(expopdf[0:50],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('pdf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')

plt.figure(3)
cal3,expocdf = a.cdf_arrival()
plt.plot(cal3[0:200],'bo-',label='Calculated')
plt.plot(expocdf[0:200],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('cdf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')

T = 5
lamda = 1
n = 10**2
times_run = 10**5

#Question A
plt.figure(4)
a = Subway(T,lamda,n,times_run)
cal,ppdf = a.Calculate_Poisson_PMF()
plt.plot(cal[0:50],'bo-',label='Calculated')
plt.plot(ppdf[0:50],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('pmf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')


plt.figure(5)
cal2,expopdf = a.pdf_arrival()
cal2[0] = 1
plt.plot(cal2[0:50],'bo-',label='Calculated')
plt.plot(expopdf[0:50],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('pdf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')


plt.figure(6)
cal3,expocdf = a.cdf_arrival()
plt.plot(cal3[0:200],'bo-',label='Calculated')
plt.plot(expocdf[0:200],'ro-',label='Estimated')
plt.xlabel('t')
plt.ylabel('cdf')
plt.title("$\lambda$ = %s, T = %s, n = %s, p =%s"%(lamda,T,n,a.p))
plt.grid()
plt.legend(loc='best')

plt.show()
