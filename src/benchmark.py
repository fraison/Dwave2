import numpy as np

import logging as logger
import Utils as utils
import json

logger.basicConfig( level=logger.INFO)
command='python ./src/benchmark.py'

class Inputs():
  def __init__(self):
      """
        Initialize input parameters for the regression tests.
        
        Attributes:
        -----------
        d : int
            Number of parameters or unknowns (excluding b).
        n : int
            Number of equations. If n <= d, the system is undetermined as b is not counted.
        xt : ndarray
            Input feature matrix.
        params : ndarray
            True parameters used to generate target values.
        b : float
            Bias term can be considered as an additional parameter set to 1
        sigma : float
            Noise level for the target values.
        full_params : ndarray
            Complete parameter vector including bias.
        pDec : int
            Precision value.
        yt : ndarray
            Generated target values.
        P : ndarray
            Precision vector for BQM binary.
      """
       
      # bias
      self.b = 1.0
      
      #noise
      self.sigma = 0.#0.2   
      
      #factor for setting the upper bound for the integer variables.
      self.fact_bound = 1.5
        
      # precision vector decimal 
      self.pDec = 10
      
      #precision vector BQM binary
      self.P = np.array([0.1,0.2,0.4,0.8,1.6,3.2,6.4]).T
      
      
      
def saveResults(classic_delta_ts, allLoopParams, iteration):
    """
      Save running time and resulting parameters to files.
        
      Attributes:
      -----------
      classic_delta_ts : numpy array
          Running times (n, m) with n is the number of free parameters lists and m is the number of regression functions used.
      allLoopParams : list of list
          each sublist contains the input full parameters and then the recovered parameters by each function. There are d sublists, one for each value of
      iteration :
          number to distinguish the result files 
    """
    logger.info("save running times")
    np.savetxt("./times_"+iteration+".csv", classic_delta_ts , delimiter=",")

    logger.info("save params")
    with open("./params_"+iteration+".json", 'w') as f:
        json.dump(allLoopParams, f, indent=2) 
        

def numberOfEquations(d, overdetermined):
    """
      Calculates the number of equations wrt number of parameters.
        
      Attributes:
      -----------
      d : int
        Degree of the polynomial or number of features minus one. free parameters plus bias term b give a total of d+1 parameters
      overdetermined : boolean
          is the system overdetermined or underdetermined
          
      Returns:
       --------
      integer
          number of equationsto generate
    """

    if overdetermined:
        # overdetermined case: here the number of equations (or data points) is 2x the number of parameters
        n = int(2*d)
    else:
        n = int(np.floor(d/2))
    
    return n    
        
        
       
def runBenchmark(regularization, regFuncs, overdetermined, iteration):
    """
      launch the benchmarking.
        
      Attributes:
      -----------
      regularization : boolean
        use regularization or not
      overdetermined : boolean
          is the system overdetermined or underdetermined
       iteration :
          number to distinguish the result files
    """

    # get input parameters
    inp = Inputs()
    
    
    #number of different functions 
    m = len (regFuncs)


    #LOOP
    # list of the numbers of variables (dimensions) or free parameters
    #r = [1, 3] #don't expect good result here with sigma != 0
    r = [2,4]
    #r = [4,8,16,32,64, 128, 256, 512]
    
    #STORE RESULTS
    classic_delta_ts = np.zeros(shape=(len(r), m+1)) 
    allLoopParams=[]

    #loop on dimensions
    i=0
    
    # d free parameters plus bias term b give a total of d+1 parameters
    # there are d parameters, excluding b 
    for d in r:
        loopParams=[]
        # len(X) <d+1 (intercept)
        # n: number of equations or data points
        # d free parameters plus bias term b give a total of d+1 parameters
        # n equations; if n<=d, system is undetermined . If n=d+1, there is a unique solution when using 'utils.gt'.
        
        n = numberOfEquations(d, overdetermined)
        logger.info("n:{}".format(n))
        
        # Input feature matrix with n equations of d parameters
        # data points x
        xt = 10*np.random.random((n,d))
        
        # parameters to recover 
        params = 10*np.random.random(d)
        at = params.reshape(d,1)
        
        full_params = np.append(params, inp.b)
        classic_delta_ts[i,0] = d 
        logger.info("------------------")
        logger.info("params:{}".format(full_params))
        loopParams.append(full_params.tolist()) 
    
        yt = utils.gt(xt, at, inp.b, inp.sigma)
    
        #CLASSIC1: least-squares solution using numpy linalg lstsq
        if regFuncs["classic1"]:
            delta_t, sol1= utils.classic_1(xt, yt)
            classic_delta_ts[i,1] = delta_t
            loopParams.append(sol1.tolist())
        else:
            #all results set to zero if can't get values
            classic_delta_ts[i,1] = 0.
            psol = np.zeros(len(full_params))
            loopParams.append(psol.tolist())    
            
        # CLASSIC2: least squares minimization using the Nelder-Mead method WITHOUT regularization.
        # Note that it needs a guess, based completely on full_params here
        if regFuncs["classic2"]:
            delta_t, sol2 = utils.classic_2(xt, yt, inp.pDec , d, full_params, utils.funT2)
            classic_delta_ts[i,2] = delta_t
            loopParams.append(sol2.tolist())
        else:
            #all results set to zero if can't get values
            classic_delta_ts[i,2] = 0.
            psol = np.zeros(len(full_params))
            loopParams.append(psol.tolist())    
            
        #HYBRID
        # Note that it needs an upper bound that we take as b x full_params here
        if regFuncs["hybrid1"]:
            delta_t, sol3 = utils.hybrid_1_reg(xt, yt, inp.pDec, d, inp.fact_bound, full_params, regularization = regularization) #only pure regression
            classic_delta_ts[i,3] = delta_t  
            loopParams.append(sol3.tolist())            
        else:
            #all results set to zero if can't get values
            classic_delta_ts[i,3] = 0.
            psol = np.zeros(len(full_params))
            loopParams.append(psol.tolist())   
            

        #BQM    
        if regFuncs["exactBQM"] or regFuncs["quantumBQM"]:
            bqm, PP = utils.makeBQM(xt, yt, inp.P, d)
            
        
        #BQM CLASSIC (classic solver)
        if regFuncs["exactBQM"]:
            delta_t, sol4 = utils.exactSolFromBQM(bqm, PP, len(full_params))
            classic_delta_ts[i,4] = delta_t  
            loopParams.append(sol4.tolist())
        else:
            classic_delta_ts[i,4] = 0.
            psol = np.zeros(len(full_params))
            loopParams.append(psol.tolist())            
                        
        #BQM QUANTUM (pure quantum)
        if regFuncs["quantumBQM"]:
            delta_t, sol5 = utils.quantumSolFromBQM(bqm, PP, len(full_params))
            classic_delta_ts[i,5] = delta_t
            loopParams.append(sol5.tolist())
        else:
            classic_delta_ts[i,5] = 0
            psol = np.zeros(len(full_params))
            loopParams.append(psol.tolist())
        
        allLoopParams.append(loopParams)
        i=i+1


    #logger.info("time table:{}".format(classic_delta_ts))    
    #logger.info("Params: {}".format(allLoopParams))
    
    
    # STORE Results
    saveResults(classic_delta_ts, allLoopParams, iteration)
    
    
    
    
def mainMethod():

    # case 
    overdetermined = False
    

    if overdetermined:

        # OVERDETERMINED CASE  
        
        # iteraton added to the result filenames; number to distinguish the result files
        iteration = "1"
    
        #SETUP
        regularization = False
        
        # calls of functions for regression
        regFuncs = {"classic1":True, "classic2":True, "hybrid1":True, "exactBQM":True, "quantumBQM":True}

        # run benchmark
        runBenchmark(regularization, regFuncs, overdetermined, iteration)


    else:
        
        # UNDERDERTERMINED CASE
        
        # iteraton added to the result filenames; number to distinguish the result files
        iteration = "2"
        
        #SETUP
        regularization = True
        
        # calls of functions for regression
        # only hybdrid here because my benchmark asks for too much resources
        regFuncs = {"classic1":True, "classic2":True, "hybrid1":True, "exactBQM":False, "quantumBQM":False}

        # run benchmark
        runBenchmark(regularization, regFuncs, overdetermined, iteration)       
        
    
          
if __name__ == '__main__':

    mainMethod()
     


