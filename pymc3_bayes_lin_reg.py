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

    print(az.summary(samples, round_to=2))
