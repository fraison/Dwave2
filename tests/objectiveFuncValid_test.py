import pytest
import numpy as np

import logging as logger
import src.Utils as utils



logger.basicConfig( level=logger.INFO)
command='pytest -v objectiveFuncValid_test.py'

class Inputs():
  def __init__(self):
   
      # d parameters or unknowns (excluding b)
      self.d = 2 # total 3 parameters

      # data points x
      self.xt = np.array([[9.,4.],  [5.,7.]])
      
      # n equations; if n<=d , system is undetermined 
      self.n = len(self.xt)
      # parameters
      self.params = np.array([3.,8.,])  
      self.b = 1.0
      self.sigma = 0# 0.2
      self.full_params = np.append(self.params, self.b)
      # precision of parameters: solution needs de-scaling
      self.pDec = 1.
      # data points y
      self.at = self.params.reshape(self.d,1)
      self.yt = utils.gt(self.xt,self.at,self.b,self.sigma)
      #precision vector BQM binary
      self.P=np.array([0.1,0.2,0.4,0.8,1.6,3.2,6.4]).T
      self.xxt=np.vstack([self.xt.T, np.ones(len(self.xt))]).T

@pytest.fixture(scope="class")
def getInputs():
    myInputs = Inputs()
    return myInputs



class Inputs2():
  def __init__(self):
   
      # d parameters or unknowns (excluding b)
      self.d = 3 # total 4 parameters

      # data points x
      self.xt = np.array([[9.,4.,2.],  [5.,7.,8.],  [3.,7.,6.]])
      
      # n equations; if n<=d , system is undetermined
      self.n = 4
      
      # parameters
      self.params = np.array([3.,8.,5.])  
      
      
      self.b = 1.0
      self.sigma = 0# 0.2
      self.full_params = np.append(self.params, self.b)
      # precision of parameters: solution needs de-scaling
      self.pDec = 1.
      # data points y
      self.at = self.params.reshape(self.d,1)
      self.yt = utils.gt(self.xt,self.at,self.b,self.sigma)
      #precision vector BQM binary
      self.P=np.array([0.1,0.2,0.4,0.8,1.6,3.2,6.4]).T
      self.xxt=np.vstack([self.xt.T, np.ones(len(self.xt))]).T

@pytest.fixture(scope="class")
def getInputs2():
    myInputs = Inputs2()
    return myInputs

class Test_regression(object):    


    
    
    def test_objective_1(self, getInputs):
        """
        Test the funT2( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs
        
        
        of1 = utils.funT2( tuple(np.floor(tIn.full_params)), tIn.xxt, tIn.yt ,tIn.pDec , tIn.d)


        logger.info("objective function result:{}".format(of1))
        
        # if p=1 we should have a null objective function

        diff1 = of1[0] - 0.0

        
        tol = 0.
        
        logger.info("difference between calculated objective function and expected value is :"+str(diff1)+" and should be less or equal than: "+str(tol))
        assert (np.abs(diff1) <= tol)
        


         
    def test_objective_2(self, getInputs):
        """
        Test the funT2_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs
        
        
        of1 = utils.funT2_reg( tuple(np.floor(tIn.full_params)), tIn.xxt, tIn.yt ,tIn.pDec , tIn.d)


        logger.info("regularized  objective function result:{}".format(of1))
        
        # if p=1 we should have a null objective function

        diff1 = of1[0] - 0.0

        
        tol = 0.
        
        logger.info("difference between calculated regularized objective function and expected value is :"+str(diff1)+" and should be less or equal than: "+str(tol))
        assert (np.abs(diff1) <= tol)
        
   
    def test_objective_3(self, getInputs):
        """
        Test the funT2_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs
        
        
        of1 = utils.funT2_reg_opt( tuple(np.floor(tIn.full_params)), tIn.xxt, tIn.yt ,tIn.pDec , tIn.d)


        logger.info("regularized & vectorized objective function result:{}".format(of1))
        
        # if p=1 we should have a null objective function

        diff1 = of1 - 0.0

        
        tol = 0.
        
        logger.info("difference between calculated regularized & vectorized objective function and expected value is :"+str(diff1)+" and should be less or equal than: "+str(tol))
        assert (np.abs(diff1) <= tol)
        
        
         
    def test_objective_4(self, getInputs2):
        """
        Test the funT2_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs2
        
        
        of1 = utils.funT2_reg( tuple(np.floor(tIn.full_params)), tIn.xxt, tIn.yt ,tIn.pDec , tIn.d)


        logger.info("regularized  objective function result:{}".format(of1))
        
        # if p=1 we should have a null objective function

        diff1 = of1[0] - 0.0

        
        tol = 0.
        
        logger.info("difference between calculated regularized objective function and expected value is :"+str(diff1)+" and should be less or equal than: "+str(tol))
        assert (np.abs(diff1) <= tol)
        
   
    def test_objective_4(self, getInputs2):
        """
        Test the funT2_reg( function.

        Parameters:
        -----------
        getInputs : Inputs
            Input parameters for the test.
        """     
             
     
        tIn = getInputs2
        
        
        of1 = utils.funT2_reg_opt( tuple(np.floor(tIn.full_params)), tIn.xxt, tIn.yt ,tIn.pDec , tIn.d)


        logger.info("regularized & vectorized objective function result:{}".format(of1))
        
        # if p=1 we should have a null objective function
        c = tIn.full_params
        target = np.sum((c[2:tIn.d] - 2 * c[1:tIn.d-1] + c[0:tIn.d-2]) ** 2)
        diff1 = of1 - target

        
        tol = 0.
        
        logger.info("difference between calculated regularized & vectorized objective function and expected value is :"+str(diff1)+" and should be less or equal than: "+str(tol))
        assert (np.abs(diff1) <= tol)
        
        
                
        
        

