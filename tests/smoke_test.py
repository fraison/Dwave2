import pytest
import numpy as np

import logging as logger
import src.Utils as utils



logger.basicConfig( level=logger.INFO)
command='pytest -v tests/smoke_test.py'

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
   
      # d parameters or unknowns (excluding b)
      self.d = 2 # d free parameters plus bias term b give a total of d+1 parameters

      # data points x
      self.xt = np.array([[9., 4.], [5., 7.],[3., 2.]])
      
      # n equations; if n<=d, system is undetermined . If n=d+1, there is a unique solution when using 'utils.gt'.
      self.n = len(self.xt)
      
      # parameters
      self.params = np.array([3., 8.])  
      
      # bias
      self.b = 1.0
      
      #factor for setting the upper bound for the integer variables.
      self.fact_bound =2.0
      
      #noise
      self.sigma = 0# 0.2
      
      # parameters with bias
      self.full_params = np.append(self.params, self.b)
      
      # precision 
      self.pDec = 10
      
      # data points y
      self.at = self.params.reshape(self.d,1)
      self.yt = utils.gt(self.xt,self.at,self.b,self.sigma)
      
      #precision vector BQM binary
      self.P=np.array([0.1,0.2,0.4,0.8,1.6,3.2,6.4]).T
      

@pytest.fixture(scope="class")
def getInputs():
    myInputs = Inputs()
    return myInputs




class Test_regression(object):    


    
    
    def test_classic_1(self, getInputs):
        """
        Test the classic_1( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs

        delta_t, sol1 = utils.classic_1(tIn.xt, tIn.yt)
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.at[0]
        diff2 = sol1[1] - tIn.at[1]
        diff3 = sol1[2] - tIn.b
        
        tol = 1e-14
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
        
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol)


    
    def test_classic_2(self, getInputs):
        """
        Test the classic_2( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
        

        tIn = getInputs

        delta_t, sol1 = utils.classic_2(tIn.xt, tIn.yt, tIn.pDec ,tIn.d, tIn.full_params, utils.funT2)
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 1e-4
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol) 
    
        
    def test_classic_2_reg(self, getInputs):
        """
        Test the classic_2( function. Here regularization has no impact as np.sum((c[2:d] - 2 * c[1:d-1] + c[0:d-2]) ** 2) =0 with getInputs

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
        

        tIn = getInputs

        delta_t, sol1 = utils.classic_2(tIn.xt, tIn.yt, tIn.pDec ,tIn.d, tIn.full_params, utils.funT2_reg)
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 1e-4
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol)
        
        
        
    
    
    def test_hybrid(self, getInputs):
        """
        Test the hybrid_1_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
        
        tIn = getInputs
     
        delta_t, sol1 = utils.hybrid_1_reg( tIn.xt, tIn.yt, tIn.pDec, tIn.d, tIn.fact_bound, tIn.full_params)
    
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 1e-20 #at most
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol) 
        
    
    
    
    def test_hybrid_noRegularization(self, getInputs):
        """
        Test the hybrid_1_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
        
        tIn = getInputs
     
        delta_t, sol1 = utils.hybrid_1_reg( tIn.xt, tIn.yt, tIn.pDec, tIn.d, tIn.fact_bound, tIn.full_params, regularization = False)
    
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 1e-20 #at most
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol) 
        
        
        
    def test_exactSolFromBQM(self, getInputs):
        """
        Test the exactSolFromBQM function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """
    
        tIn = getInputs
        
        bqm, PP = utils.makeBQM(tIn.xt, tIn.yt, tIn.P, tIn.d)
        
        #BQM CLASSIC
        delta_t, sol1 = utils.exactSolFromBQM(bqm, PP, len(tIn.full_params))
               
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 1e-20
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        logger.info("difference between generated and reference parameter3 is :"+str(diff3)+" and should be less than: "+str(tol))
        assert (np.abs(diff3) < tol) 
        
        
    
    def test_quantumSolFromBQM(self, getInputs):
        """
        Test the quantumSolFromBQM function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """
    
        tIn = getInputs
        
        bqm, PP = utils.makeBQM(tIn.xt, tIn.yt, tIn.P, tIn.d)
        
        #BQM CLASSIC
        delta_t, sol1 = utils.quantumSolFromBQM(bqm, PP, len(tIn.full_params))
               
        logger.info("sol:{}".format(sol1))

        diff1 = sol1[0] - tIn.full_params[0]
        diff2 = sol1[1] - tIn.full_params[1]
        diff3 = sol1[2] - tIn.full_params[2]
        
        tol = 0.6
        
        
        logger.info("difference between generated and reference parameter1 is :"+str(diff1)+" and should be less than: "+str(tol))
        assert (np.abs(diff1) < tol)
        
        logger.info("difference between generated and reference parameter2 is :"+str(diff2)+" and should be less than: "+str(tol))
        assert (np.abs(diff2) < tol)
                
        #WARNING bias is not supposed to be recovered
        
        
    
        
        
        
        
        
        
        

