import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from generate_data import *
import arviz as az


dist_sigma = 0.2   #Precision (inverse variance) = 1/sigma**2 = 25, inverse std = 1/sigma = 5
dist_mean = 0
N = 100
vals = np.array([-0.3,0.5])

gen_dat(N, dist_sigma, dist_mean, vals)
path = "./Datasets/"
data = "GaussianData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(dist_sigma)+".txt"

data = np.loadtxt(path+data)
x = data[:,0]
y = data[:,1]

my_model = pm.Model()
alpha0 = 2*10**(3)      #Standard deviation of prior over w0
alpha1 = 5*10**(3)      #Standard deviation of prior over w1
tau = 0.2               #Variance of prior over beta (inverse standard deviation of the likelihood function)

with my_model:
    print("Defining priors...")

    # Priors for unknown model parameters
    w0 = pm.Normal("w0", mu=0.0, sd=alpha0)
    w1 = pm.Normal("w1", mu=0.0, sd=alpha1)
    beta = pm.HalfNormal("beta", tau=tau)   #Normal distribution bound at zero (variance strictly positive)


    # Expected value of outcome
    mu = w0 + w1*x

    print("Defining likelihood...")
    # Likelihood (sampling distribution) of observations
    likelihood_f = pm.Normal("Likelihood function", mu=mu, sd=beta**(-1), observed=y)
    print("Model initiated!")

    # Map estimate
    map_estimate = pm.find_MAP()
    print("Map estimate:")
    print(map_estimate)
    print(" ")

    #Sampling using Hamiltonian No-U-Turns
    print("Starting HMC-NUTS sampling...")
    samples = pm.sample(draws=2000, cores = 1,return_inferencedata=True)
    az.plot_trace(samples)
    plt.savefig("Pymc3_plot.pdf")
    plt.show()

    #az.plot_posterior(samples)
    print(az.summary(samples, round_to=2))





"""
Starting HMC-NUTS sampling...
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Sequential sampling (2 chains in 1 job)
NUTS: [beta, w1, w0]
Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 3 seconds.0% [3000/3000 00:01<00:00 Sampling chain 1, 0 divergences]
      mean    sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
w0   -0.31  0.02   -0.34    -0.27       0.00      0.0   4890.39   2795.00    1.0
w1    0.50  0.04    0.43     0.56       0.00      0.0   5204.00   3025.90    1.0
beta  5.04  0.36    4.31     5.69       0.01      0.0   4765.59   2647.03    1.0
"""
