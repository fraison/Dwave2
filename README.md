# Dwave2
Benchmarking of linear regression based on least squares with or without regularization using classical and quantum methods (hybrid and QUBO).


I) Motivation: 
The motivation of this project is to explore methods to accelerate calculations for deprojection and/or galaxy model parameters determination in astronomy. 
These use cases typically involve a very resource intensive multidimensional least square regression with regularization lasting a couple of days on a standard machine. In the case of deprojection, the regression should determine the values of up to 400 parameters from 200 points. And an approximate solution would be already very usefull to reduce the time it takes to find a very accurate parameter set. Within this use case, a specific form of regularization is chosen in order to guaranty the smoothness of the solution.
These benchmarks aim at comparing 2 Python Numpy functions run on a classical computer with different quantum computing methods. In more details, the quantum approach is performed with functions running at least partly on a Dwave quantum annealer (hybrid, QUBO) or running on the local CPU in the case of the QUBO solver used normally for testing and debugging code. 
The formulation of the QUBO error function is based on Date and Potok (see reference) which largely inspired this work. This code can be used to do a scalability study by performing a least square regression with an increasing number of parameters. Additionnally, the regression error can be studied.  The benchmarks can solve an underdetermined system of n equations with 2n paramters as in the use case but also an overdetermined system with n parameters and 2n equations in order to see the impact of the number of points. 


Notes: 
0) data generation:
Data are generated randomly and noise is added to the data to make it more realistic. 

1) recovery of bias:
The static bias (here b) in a BQM is a fixed offset that does not affect the optimization process. It cannot be recovered from the solution because it is not variable-dependent and does not influence the relative energy levels of different configurations. It is specified as part of the BQM and remains constant throughout the process. Consequently, it is set to one.

2) precision vector or factor: 
As the hybrid quantum method returns integer values, we get the requested precision by multiplying the initial equation by a precision fatorcr and need to divide the return values by the same factor to get the correct parameters. In the code, a factor 10 was implemented to get a precision of 1/10. But a meaningfull benchmarking must be done with the precision corresponding to each specific use case. With QUBO, only binaries values can be returned. We used the principles described in Date and Potok article. 


2) limits of QUBO in numbers of parameters:
QUBO didn't accept the large numbers of parameters that were submitted especially with the underdetermined cases. Consequently, the results were avaialble either for up to the order of 10 parameters  (overdetermined case) or none (underdetermined). 

3) QUBO doesn't return an exact solution:
The regression is expressed as a QUBO problem. And adiabatic quantum computer solve QUBO approximatively. 


4) Processing time:
Regarding quantum method using D-Wave, the calculation time refers to the annealing time which is returned with the results. It is much smaller than the total processing time but more stable as it doesn't take the network into account. Additionnally, it is more comparable to the Python "Time" method as the Python code is running on the same machine.  


5)  Source of errors with D-Wave
Inter qbit connections can break.
        



II) Reference:
Date, P., Potok, T. Adiabatic quantum linear regression. Sci Rep 11, 21905 (2021). https://doi.org/10.1038/s41598-021-01445-6
