

import numpy as np
import json
import matplotlib.pyplot as plt
import logging as logger
import copy

logger.basicConfig( level=logger.INFO)
command='python ./src/analysis.py'


# analysis running time

times_array = np.loadtxt('./times_2.csv', delimiter=',')


print(times_array)

#pyplot
plt.plot(times_array[:,0], times_array[:,1], 'rs', label='np.linalg.lstsq')
plt.plot(times_array[:,0], times_array[:,2], 'bs', label='scipy.optimize.minimize')
plt.plot(times_array[:,0], times_array[:,3], 'g^', label='Hybrid')
plt.plot(times_array[:,0], times_array[:,4], 'gs', label='BQM exact')
plt.plot(times_array[:,0], times_array[:,5], 'b^', label='BQM quantum')
plt.title("Exec time fo different functions with Npoints = Nparams/2; prec=10")
plt.yscale('log')
plt.xlabel('number of parameters: Nparams ')
plt.ylabel('time [ms]')
plt.legend(loc='upper left')
#plt.savefig("batchQuantum1.png")
plt.show()

#------------------------------------------------------

#analysis  parameters precision

def getJsonRes(filename):
    """
    Read json file
    """
    with open(filename, 'r') as f:
        MML=json.load(f)

    return MML




pall = getJsonRes("params_2.json")


# number of functions
m=6

ptable = np.zeros(shape=(len(pall), m)) 

for k,l in enumerate(pall):
    ptable[k][0] = len(l[0])
    ref = copy.copy (l[0]) #shallow
    # calculate distance of each set of resulting parameters fom reference set
    q=1
    for p in l[1:]:#a list of param
        if np.sum(np.abs(np.array(p)))>0:
            #distance = sum of absolute difference between each parameter and corresponding parameter from reference set
            s= np.sum([np.abs(np.array(i)-np.array(j)) for i,j in zip(ref, p)])
        else:
            s = 0 # no retrun from a called function
        ptable[k][q] = s/len(ref) #normalize
        print(k,q,s)
        q=q+1



plt.plot(ptable[:,0], ptable[:,1], 'rs', label='np.linalg.lstsq')
plt.plot(ptable[:,0], ptable[:,2], 'bs', label='scipy.optimize.minimize regul')
plt.plot(ptable[:,0], ptable[:,3], 'g^', label='Hybrid regul')
#plt.plot(ptable[:,0], ptable[:,4], 'gs', label='BQM exact')
#plt.plot(ptable[:,0], ptable[:,5], 'b^', label='BQM quantum')
plt.title("Normalized Error (sum of abs differences) for different functions with Npoints=Nparams/2; prec=10")
plt.yscale('log')
plt.xlabel('number of parameters ')
plt.ylabel('error')
plt.legend(loc='upper right')
plt.savefig("batchQuantumErr1.png")
plt.show()


