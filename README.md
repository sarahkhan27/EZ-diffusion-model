# EZ-diffusion-model
Final Cogs 106 Assignment - EZ Diffusion Model (Simulate and Recover)

A goal for this project was to determine the functionality of the EZ diffusion model (cognitive model) when testing on estimate parameters generated from the EZ diffusion model itself by using a simulate and recover method. 

I began by setting up my EZ-diffusion-model repository which consists of folders and files named src (contains all the main code files), test (contains all the test files), results (saves all the test outputs as txt files), src/simulate_and_recover.py (houses the EZ equations - forward EZ for simulating summary statistical predictions, inverse EZ for recovering observed statistical parameters), src/simulation_main.py, src/main.sh, test/test_simulation.py (runs tests), and test/test.sh. 

A requirement was to run 1000 stimulation tests per sample size (N=10, N=40, and N=400) which outputs 3000 values. The EZ diffusion model also analyzes decision making processes via drift rate (v), boundary separation (a), and non-decision time (t) within specific ranges (proper ranges to use defined in the canvas assignment page). Each of the tests ran on these parameters and codes also check for various qualities such as varifications if the expectced trend is occuring between bias and sample size, correct EZ equation implamentation, and more. 

EZ Diffusion Model Simulation Results (these can also be found in the results folder):
===================================

Sample size N = 10
Average bias for v: -0.238210
Average bias for alpha: -0.200770
Average bias for tau: 0.004627
Average squared error: 1.161530

Sample size N = 40
Average bias for v: -0.047913
Average bias for alpha: -0.033970
Average bias for tau: 0.002890
Average squared error: 0.196038

Sample size N = 4000
Average bias for v: -0.000004
Average bias for alpha: 0.000109
Average bias for tau: 0.000161
Average squared error: 0.001304

Something that is critical to note and observed in the results is the bias value (difference between the true parameter values and the estimated ones) for each of the three parameters. The bias is close to 0 initially and reduces as the sample size increases since the variance also reduces with increasing size (this trend is also noted in the squared error). These key points and valuable trends support the proper functionality of the EZ diffusion model, as the sample size increases, in terms of recovering parameters. This highlights the improvment and accuracy of recovery with increasing sample size and reduction of bais and error, which is expected behavior hypothesized. 