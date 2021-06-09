import numpy as np
import os
np.random.seed(1001)

def gen_dat(N,sigma,mu, vals):

    x = np.random.uniform(-1,1, N)

    def f(x,coeffs):
        f_out = 0
        for i in range(len(coeffs)):
            f_out += coeffs[i]*x**i
        return f_out

    noise = np.random.normal(mu, sigma, size = N)

    data = f(x,vals) + noise

    filename = "GaussianData_N_" + str(N) + "_mean_" + str(mu) + "_sigma_" + str(sigma) + ".txt"

    if len(vals) != 2:
        filename = "GaussianData_N_" + str(N) + "_mean_" + str(mu) + "_sigma_" + str(sigma) + "_p_"+str(len(vals)) +".txt"


    with open(filename,"w") as outfile:
        for i in range(N):
            outfile.write(str(x[i]) + " " + str(data[i]))
            outfile.write("\n")

    path = "./Datasets/"
    if not os.path.exists(path):
        os.makedirs(path)

    os.system("mv" + " " + filename + " " + path)


def gen_dat_sin(N,sigma,mu):

    x = np.random.uniform(-1,1, N)

    def f(x):
        return np.sin(2*np.pi*x)

    noise = np.random.normal(mu, sigma, size = N)

    data = f(x) + noise

    filename = "SinData_N_" + str(N) + "_mean_" + str(mu) + "_sigma_" + str(sigma) + ".txt"

    with open(filename,"w") as outfile:
        for i in range(N):
            outfile.write(str(x[i]) + " " + str(data[i]))
            outfile.write("\n")

    path = "./Datasets/"
    if not os.path.exists(path):
        os.makedirs(path)

    os.system("mv" + " " + filename + " " + path)
