import numpy as np
import os
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn import linear_model
import pymc3 as pm
import arviz as az

from generate_data import *
from bayesian_linear_regression import *

produce_data = sys.argv[1]

sigma = 0.2
alpha = 5*10**(-3)
dist_mean = 0
N = 100

path = "./Datasets/"
beta = (1/sigma)**2

x = np.linspace(-1,1,1000)

vals = np.array([-0.3,0.5])
fz = 14


if produce_data == "gauss":
    # Ser at lavere alpha gir posterior vektfordeling nærmere ekte punkt enn høy
    # Beta langt unna ekte gir større fordeling
    N = [0,2,10,100]
    for n in N:
        #gen_dat(n,sigma, dist_mean, vals)
        data = "GaussianData_N_"+str(n)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
        degree = 2    #Should be number of parameters INCLUDING bias term
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean,produce_data,vals,deg=degree)
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        my_solver.plot_data_space(x)
        my_solver.plot_posterior()




if produce_data == "hyppar":
    n = 50
    gen_dat(n,sigma, dist_mean, vals)
    data = "GaussianData_N_"+str(n)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
    degree = 2    #Should be number of parameters INCLUDING bias term
    alpha_list = [alpha, alpha*10,  alpha*100, alpha*1000]
    beta_list = [beta, beta*10, beta/10, beta/100]
    for a in alpha_list:
        for b in beta_list:
            my_solver = Bayesian_Linear_Regression(b,a, dist_mean,produce_data,vals,deg=degree)
            my_solver.read_data(path + data)
            my_solver.posterior_f()
            my_solver.plot_data_space(x)
            my_solver.plot_posterior()


if produce_data == "sin":
    degree = 10    #Should be number of parameters INCLUDING bias term
    N = [0,2,5,10,50,100]
    for n in N:
        gen_dat_sin(n,sigma, dist_mean)
        data = "SinData_N_"+str(n)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean,produce_data, vals,deg=degree)
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        my_solver.plot_sinoidal(x)

if produce_data == "poly":
    gen_dat_sin(N,sigma, dist_mean)
    data = "SinData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
    fig, ax = plt.subplots(2,5,constrained_layout=True,sharex=True, sharey=True)
    ax = ax.ravel()
    degs = [1,2,3,4,5,6,7,8,9,10]
    marg_liks = np.zeros(len(degs))
    for i in range(len(degs)):
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean, produce_data, vals,deg = degs[i])
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        my_solver.plot_many_poly(x,ax[i])
        marg_liks[i] = my_solver.log_marg_likelihood()

    #fig.tight_layout()
    fig.supxlabel('x', fontsize = 16)
    fig.supylabel('y', fontsize = 16)
    #fig.suptitle("Real data = %i points" % N)
    fig.suptitle("Polynomial basis functions of degree p", fontsize = 16)
    plt.show()

    degs = np.array(degs)
    plt.plot(degs-1,marg_liks)
    plt.xlim(0,degs[-1]-1)
    plt.xlabel("Order of polynomial", fontsize = 16)
    plt.ylabel("Model evidence", fontsize = 16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("./Results/ModelEvidence.pdf")
    plt.show()

if produce_data == "poly_test":
    vals = [1,2*10**(-1),-3,4,5,6,7,8,9]
    gen_dat(N,sigma, dist_mean, vals)
    data = "GaussianData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+ "_p_"+str(len(vals)) +".txt"
    fig, ax = plt.subplots(3,4)
    ax = ax.ravel()

    degs = [1,2,3,4,5,6,7,8,9,10]
    marg_liks = np.zeros(len(degs))
    for i in range(len(degs)):
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean, "poly",vals, deg = degs[i])
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        my_solver.plot_many_poly(x,ax[i])
        marg_liks[i] = my_solver.log_marg_likelihood()

    fig.tight_layout()
    plt.show()

    degs = np.array(degs)
    plt.plot(degs,marg_liks, "*", markersize = 14 )
    plt.xlabel("Order of polynomial")
    plt.ylabel("Model evidence")
    plt.show()

if produce_data == "sin_many":
    degs = [1,2,3,4,5,6,7,8,9,10]
    gen_dat_sin(N,sigma, dist_mean)
    data = "SinData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
    for i in range(len(degs)):
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean,"sin",vals, deg=degs[i])
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        my_solver.plot_sinoidal(x)

if produce_data == "max_evidence":
    tol = 10**(-5)
    degree = 10
    alpha = 1
    beta = 1
    alphas = []
    betas = []
    gen_dat_sin(N,sigma, dist_mean)
    data = "SinData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
    for iteration in range(1000):
        alphas.append(alpha)
        betas.append(beta)
        #print("alpha = ",alpha, "beta = ", beta)
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean, "poly",vals, deg = degree)
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        new_alpha, new_beta = my_solver.max_evidence()

        if np.abs(alpha - new_alpha) < tol and np.abs(beta-new_beta) < tol:
            alphas.append(new_alpha)
            betas.append(new_beta)
            alphas = np.array(alphas)
            betas = np.array(betas)
            itt = np.array(list(range(0, iteration+2)))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Iterative convergence for $\\alpha$ and $\\beta$', fontsize = 16)
            ax1.plot(itt, alphas,"*")
            ax1.plot(itt[-1], alphas[-1],"*", label= "Optimal $\\alpha$ = %.4f" % alphas[-1])
            ax1.set_ylabel("$\\alpha$", fontsize = 16)
            ax2.plot(itt, betas,"*")
            ax2.plot(itt[-1], betas[-1],"*", label= "Optimal $\\beta$ = %.4f" %betas[-1])
            ax2.set_ylabel("$\\beta$", fontsize = 16)
            fig.supxlabel('Iterations', fontsize = 16)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax2.tick_params(axis='both', which='major', labelsize=14)
            ax1.legend()
            ax2.legend()
            plt.show()
            my_solver.plot_max_evidence(x)
            break

        alpha, beta = new_alpha, new_beta

if produce_data == "skl":
    N = 100
    tol = 10**(-5)
    degree = 10
    alpha = 1
    beta = 1
    alphas = []
    betas = []
    gen_dat_sin(N,sigma, dist_mean)
    data = "SinData_N_"+str(N)+"_mean_"+str(dist_mean)+"_sigma_"+str(sigma)+".txt"
    for iteration in range(1000):
        alphas.append(alpha)
        betas.append(beta)
        my_solver = Bayesian_Linear_Regression(beta,alpha, dist_mean, "poly",vals, deg = degree)
        my_solver.read_data(path + data)
        my_solver.posterior_f()
        new_alpha, new_beta = my_solver.max_evidence()

        if np.abs(alpha - new_alpha) < tol and np.abs(beta-new_beta) < tol:
            alphas.append(new_alpha)
            betas.append(new_beta)
            alphas = np.array(alphas)
            betas = np.array(betas)
            itt = np.array(list(range(0, iteration+2)))
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Iterative convergence for $\\alpha$ and $\\beta$', fontsize = 16)
            ax1.plot(itt, alphas,"*")
            ax1.plot(itt[-1], alphas[-1],"*", label= "Optimal $\\alpha$ = %.4f" % alphas[-1])
            ax1.set_ylabel("$\\alpha$", fontsize = 16)
            ax2.plot(itt, betas,"*")
            ax2.plot(itt[-1], betas[-1],"*", label= "Optimal $\\beta$ = %.4f" %betas[-1])
            ax2.set_ylabel("$\\beta$", fontsize = 16)
            fig.supxlabel('Iterations', fontsize = 16)
            ax1.tick_params(axis='both', which='major', labelsize=14)
            ax2.tick_params(axis='both', which='major', labelsize=14)
            ax1.legend()
            ax2.legend()
            fig.tight_layout()
            plt.savefig("./Results/Iterative_hyperparams_N_" + str(N) + ".pdf")
            plt.show()
            my_solver.plot_max_evidence(x)
            print("Weights from my model:")
            print(my_solver.current_weights)
            break

        alpha, beta = new_alpha, new_beta

    data = np.loadtxt(path+data)
    gridpoints = np.linspace(-1,1,1000)
    x = data[:,0]
    n = len(x)
    y = data[:,1]
    X = np.vander(x, degree, increasing=True)
    GP = np.vander(gridpoints, degree, increasing=True)

    skl_solver = linear_model.BayesianRidge(tol = 1e-6, fit_intercept=False, compute_score=True, alpha_init = 1.0, lambda_init = 1.0)
    skl_solver.fit(X,y)
    ymean, ystd = skl_solver.predict(GP, return_std=True)

    plt.plot(gridpoints, np.sin(2*np.pi*gridpoints), "-.",color="blue", label="Truth")
    plt.scatter(x, y, marker =  'o', alpha = 0.5, edgecolor = 'k', facecolor ='None', label="Real data = %i points" % n)
    plt.plot(gridpoints, ymean, color="red", label="Prediction mean")
    plt.fill_between(gridpoints, ymean-ystd, ymean+ystd, color="blue", alpha=0.3, zorder =0)
    plt.title("Sklearn (L = %.4f)" %skl_solver.scores_[-1], fontsize = 16)
    plt.legend(loc = "upper right")
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.0,1.0)
    plt.xlabel("x",fontsize = 14)
    plt.ylabel("y",fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig("./Results/MAXEVIDENCE_SKL_Dataspace_N_" + str(n)+".pdf")
    plt.show()


    #print(skl_solver.get_params())
    print(" ")
    print("Weights from skl:")
    print(skl_solver.coef_)
