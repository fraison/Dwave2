import numpy as np
from scipy.optimize import minimize
import time
from dimod import ConstrainedQuadraticModel, Binary, quicksum, Integer, BQM
from dimod.reference.samplers import ExactSolver
from dwave.system import LeapHybridCQMSampler
from dwave.system import DWaveSampler, EmbeddingComposite


import time


#set up
def gt(xt, a, b, sigma):
    mu = 0  # mean 
    s = np.random.normal(0, sigma, len(xt)).reshape(len(xt),1)
    return np.matmul(xt,a)+b+s
    
    

def classic_1(xt, yt):
    """
    @brief least-squares solution using numpy linalg lstsq
    @details
        This function computes least-squares solution to a linear matrix equation with classic computing using numpy linalg lstsq.
        It solves yt = m*xt+1 rewritten as yt  = xt * psol, where psol = [[x 1]]
    @details
    Parameters
    ----------
       xt : vector 
       yt : vector
    @return 
      delta_t: double 
            running time
      psol: vector
            solution of yt = xt * psol
    """
    """
    Perform a linear least squares regression using NumPy's `linalg.lstsq` method.

    This function constructs the design matrix `A`, solves for the least squares solution, and measures 
    the execution time of the process. It returns the execution time and the optimized parameters.

    Parameters:
    -----------
    xt : ndarray
        Input feature matrix.
    yt : ndarray
        Target values.

    Returns:
    --------
    tuple
        A tuple containing:
        - delta_t (float): Execution time of the least squares process in milliseconds.
        - psol (ndarray): Optimized coefficients as solution of solution of yt = xt * psol

    Notes:
    ------
    This function uses the `lstsq` function from `numpy.linalg` to solve the linear least squares problem.
    """
    A = np.vstack([xt.T, np.ones(len(xt))]).T
    try:    
       t1 = time.monotonic_ns()
       sol = np.linalg.lstsq(A, yt, rcond=None)[0]
       t2 = time.monotonic_ns()
       
       delta_t = (t2-t1)/1e6
       psol = sol.T[0]
       print("---------")
       print("run time 1 [ms]:{}".format(delta_t))
       print("classic numpy sol:{}".format(sol.T))#which one
           
    except Exception as e:
        print("Exception step 1: %s",e)
        print(type(e))    # the exception instance
        print(e.args)     # arguments stored in .args
        print(e) 
        delta_t = 0.
        psol = np.zeros(len(xt.T)+1)            
    return delta_t, psol



#classic lstsq minimization   the sum of the squares of the residuals 
def funT2(c, *args): #x,y,p = tuple
    """
    Calculate the objective function value for the least squares minimization without regularization.

    This function computes the sum of the squares of the residuals for a given set of coefficients `c`.

    Parameters:
    -----------
    c : array-like
        Coefficients for the regression model.
    args : tuple
        A tuple containing:
        - xxt (ndarray): Extended feature matrix, where the last column is a column of ones for the intercept.
        - yt (ndarray): Target values.
        - p (float): Scaling factor for the coefficients.
        - d (int): Degree of the polynomial or number of features minus one.

    Returns:
    --------
    float
        The computed value of the objective function.
    """
    xxt,yt,p,d = args[:]
    return sum([ (yt[i]*p - sum([c[j]*xxt[i][j] for j in range(d+1)]))**2 for i  in range(len(xxt)) ])
   
   

#classic lstsq minimization   the sum of the squares of the residuals 
def funT2_reg(c, *args): #x,y,p = tuple
    """
    Calculate the objective function value for the least squares minimization with regularization.

    This function computes the sum of the squares of the residuals for a given set of coefficients `c`,
    along with a regularization term that penalizes large changes between consecutive coefficients.

    Parameters:
    -----------
    c : array-like
        Coefficients for the regression model.
    args : tuple
        A tuple containing:
        - xxt (ndarray): Extended feature matrix, where the last column is a column of ones for the intercept.
        - yt (ndarray): Target values.
        - p (float): Scaling factor for the coefficients.
        - d (int): Degree of the polynomial or number of features minus one.

    Returns:
    --------
    float
        The computed value of the objective function.
    """
    xxt,yt,p,d = args[:]
    return sum([ (yt[i]*p - sum([c[j]*xxt[i][j] for j in range(d+1)]))**2 for i  in range(len(xxt)) ]) + sum([(c[j+1]-2.*c[j]+c[j-1])**2 for j in range(1,d-1)])
   
   
   
   
def funT2_reg_opt(_c, *args):
    """
    Calculate the objective function value for the least squares minimization with regularization.

    This function computes the sum of the squares of the residuals for a given set of coefficients `c`,
    along with a regularization term that penalizes large changes between consecutive coefficients. 
    The com[putation is vectorized.

    Parameters:
    -----------
    c : array-like
        Coefficients for the regression model.
    args : tuple
        A tuple containing:
        - xxt (ndarray): Extended feature matrix, where the last column is a column of ones for the intercept.
        - yt (ndarray): Target values.
        - p (float): Scaling factor for the coefficients.
        - d (int): Degree of the polynomial or number of features minus one.

    Returns:
    --------
    float
        The computed value of the objective function.
    """
    xxt, yt, p, d = args[:]
    c = np.array(_c)
    
    # Compute the residuals using matrix multiplication
    #residuals = yt * p - np.dot(xxt, c) wrong
    residuals = yt.T * p - np.dot(xxt, c) 
    
    # Compute the sum of the squares of the residuals
    sum_of_squares = np.sum(residuals ** 2)
    
    # Compute the regularization term
    #regularization = np.sum((c[1:d] - 2 * c[0:d-1] + c[0:d-2]) ** 2) wrong
    regularization = np.sum((c[2:d] - 2 * c[1:d-1] + c[0:d-2]) ** 2)
    
    return sum_of_squares + regularization   
   
   


def classic_2(xt, yt, p ,d, full_params, objFunc):
    
    """
    Perform classic least squares minimization using the Nelder-Mead method with regularization.

    This function sets up the data, initializes the optimization process, and attempts to minimize
    the objective function defined in `funT2_reg`. It measures and returns the execution time and
    the optimized parameters.

    Parameters:
    -----------
    xt : ndarray
        Input feature matrix.
    yt : ndarray
        Target values.
    p : float
        Precision as a scaling factor for the coefficients used for integer only quantum calculation
    d : int
        Degree of the polynomial or number of features minus one.
    full_params : array-like
        Initial guess for the coefficients.
    objFunc : function 
        Objective function  : funT2 or funT2_reg
    
    Returns:
    --------
    tuple
        A tuple containing:
        - delta_t (float): Execution time of the optimization process in milliseconds.
        - psol (ndarray): Optimized coefficients scaled by `p as solution of yt = xt * psol

    Notes:
    ------
    This function uses the `minimize` function from `scipy.optimize` with the Nelder-Mead method.
    """
     
    xxt=np.vstack([xt.T, np.ones(len(xt))]).T
    try:    
       t1 = time.monotonic_ns()
       #ASSUMPTION : we start from integer value of real parameters

       res2 = minimize(objFunc, tuple(np.floor(full_params)), args=(xxt, yt, p, d) , method='Nelder-Mead', tol=1e-3 )
       t2 = time.monotonic_ns()
       
       delta_t = (t2-t1)/1e6
       psol = res2.x/p
       print("---------")
       print("run time 2 [ms]:{}".format(delta_t))
       print("classic 2 scipy sol:{}".format(psol))
           
    except Exception as e:
       print("Exception step 2: %s",e)
       print(type(e))    # the exception instance
       print(e.args)     # arguments stored in .args
       print(e) 
       delta_t = 0.
       psol = np.zeros(len(xxt.T))
    
    return delta_t, psol



       
def hybrid_1_reg( xt, yt, p, d, b, params):
    """
    Perform constrained quadratic model (CQM) optimization for regression with regularization using D-Wave's hybrid solver.

    This function constructs a CQM with an objective function that includes the sum of the squares of the residuals and
    a regularization term. It then solves the problem using D-Wave's hybrid CQM solver.

    Parameters:
    -----------
    xt : ndarray
        Input feature matrix.
    yt : ndarray
        Target values.
    p : float
        Scaling factor for the coefficients.
    d : int
        Degree of the polynomial or number of features minus one.
    b : int
        Base value for setting the upper bound for the integer variables.
    params : array-like
        Initial guess for the coefficients.

    Returns:
    --------
    tuple
        A tuple containing:
        - delta_t (float): Execution time of the optimization process in milliseconds.
        - psol (ndarray): Optimized coefficients scaled by `p`.

    Notes:
    ------
    This function uses D-Wave's LeapHybridCQMSampler to solve the constrained quadratic model.
    """
 
 
    # Initialize the CQM object
    cqm = ConstrainedQuadraticModel()
    #ASSUMPTION: we set an upper bound for b
    bb =  Integer('bb', upper_bound=6*b*p)
    xxt = np.vstack([xt.T, np.ones(len(xt))]).T
    
    #------- this new one is OK
    tb=[Integer(f"aa_{i}", upper_bound=2*p*params[i]) for i in range(d+1)] #ASSUMPTION WATCH out upper bound !!
    #obj2 = quicksum([ (yt[i][0]*p - quicksum([tb[j]*xxt[i][j] for j in range(d+1)]))**2 for i  in range(len(xt)) ]) & don't use b=1 at j=d and avoid array error with j+1 and j-1
    obj2 = quicksum([ (yt[i][0]*p - quicksum([tb[j]*xxt[i][j] for j in range(d+1)]))**2 for i  in range(len(xt)) ]) + quicksum([(tb[j+1]-2.*tb[j]+tb[j-1])**2 for j in range(1,d-1)])
    
    
    cqm.set_objective(obj2) # add entropy?
    #-------
    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler()
    
    # Solve the problem using the DQM solver
    try: 
        sampleset = sampler.sample_cqm(cqm, label='Example - Trivial Regression')
        psol = np.array([v for k,v in sampleset.first.sample.items()])/p
        delta_t = sampleset.info['qpu_access_time']/1e3
        print("---------")
        print("run time 3 [ms]:{}".format(delta_t))
        print("hybrid sol:{}".format(psol))
    except Exception as e:
        print("Exception step 3: %s",e)
        print(type(e))    # the exception instance
        print(e.args)     # arguments stored in .args
        print(e)
        delta_t = 0.
        psol = np.zeros(len(xxt.T))
    return delta_t, psol
   
   



   
def makeBQM(xt, yt, P, d):
 
    """
    Construct a Binary Quadratic Model (BQM) for regression.

    This function constructs a BQM using the given feature matrix, target values, precision matrix, 
    and degree of the polynomial. It computes the bias, linear, and quadratic terms for the BQM.

    Parameters:
    -----------
    xt : ndarray
        Input feature matrix.
    yt : ndarray
        Target values.
    P : ndarray
        Precision matrix.
    d : int
        Degree of the polynomial or number of features minus one.

    Returns:
    --------
    tuple
        A tuple containing:
        - bqm2 (BQM): Constructed binary quadratic model.
        - PP (ndarray): Precision matrix used in the BQM construction.
    """
    
    PP=np.kron(np.identity(d+1),P) #precision matrix
    xxt=np.vstack([xt.T, np.ones(len(xt))]).T
    bias = yt.T@yt
    linear = -2*PP.T@xxt.T@yt
    quadratic = PP.T@xxt.T@xxt@PP
    quadratic2 = quadratic + np.diagflat(linear)
    bqm2 = BQM(quadratic2, "BINARY") #instead
    return bqm2, PP


def exactSolFromBQM(bqm, PP, size_fullp):
    """
    Solve the BQM using an exact solver and retrieve the solution.

    This function solves the given BQM using `ExactSolver` from `dimod` and calculates the solution 
    coefficients. It also measures and returns the execution time.

    Parameters:
    -----------
    bqm : BQM
        Binary quadratic model to be solved.
    PP : ndarray
        Precision matrix used in the BQM construction.
    size_fullp : int
        Size of the full parameter vector.

    Returns:
    --------
    tuple
        A tuple containing:
        - delta_t (float): Execution time of the optimization process in milliseconds.
        - psol (ndarray): Solution coefficients obtained from the BQM.
    """
    try: 
        sampler2 = ExactSolver()
        t1 = time.monotonic_ns()
        sampleset2 = sampler2.sample(bqm)
        t2 = time.monotonic_ns()
        
        res2 = np.array([v for k,v in sampleset2.first.sample.items()])
        psol = PP@res2        
        delta_t = (t2-t1)/1e6
        print("---------")
        print("run time 4 [ms]:{}".format(delta_t)) 
        print("BQM exact sol:{}".format(psol))
    except Exception as e:
        print("Exception step 4: %s",e)
        print(type(e))    # the exception instance
        print(e.args)     # arguments stored in .args
        print(e)
        delta_t = 0.
        psol = np.zeros(size_fullp) 
    return delta_t, psol  
   
   

def quantumSolFromBQM(bqm, PP, size_fullp):
    """
    Solve the BQM using a quantum annealer and retrieve the solution.

    This function solves the given BQM using D-Wave's quantum annealer and calculates the solution 
    coefficients. It also measures and returns the execution time.

    Parameters:
    -----------
    bqm : BQM
        Binary quadratic model to be solved.
    PP : ndarray
        Precision matrix used in the BQM construction.
    size_fullp : int
        Size of the full parameter vector.

    Returns:
    --------
    tuple
        A tuple containing:
        - delta_t (float): Execution time of the optimization process in milliseconds.
        - psol (ndarray): Solution coefficients obtained from the BQM.
    """
    try: 
        print("---------")
        sampler3 = EmbeddingComposite(DWaveSampler())
        print("sampling")
        sampleset3 = sampler3.sample(bqm, num_reads=2000, label='SDK Examples - Scheduling')
        print("get results")
        res3 = np.array([v for k,v in sampleset3.first.sample.items()])
        psol = PP@res3     
        delta_t = sampleset3.info['timing']['qpu_access_time']/1e3        
        print("run time 5 [ms]:{}".format(delta_t)) 
        print("BQM quantum sol:{}".format(psol))
    except Exception as e:
        print("Exception step 5: %s",e)
        print(type(e))    # the exception instance
        print(e.args)     # arguments stored in .args
        print(e)
        delta_t = 0.
        psol = np.zeros(size_fullp)
    return delta_t, psol





