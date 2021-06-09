import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps


class Bayesian_Linear_Regression:

    def __init__(self, beta, alpha, dist_mean, type,vals, deg):
        self.beta = beta
        self.alpha = alpha
        self.dist_mean = dist_mean
        self.p = deg
        self.real_weights = vals

        if type == "gauss" or type == "hyppar":
            self.phi = lambda x,i: self.phi_poly(x,i)
            self.m0 = np.zeros(self.p)   #mean of prior

        if type == "sin":
            self.phi = lambda x,i: self.phi_gauss(x,i)
            self.m0 = np.zeros(self.p)   #mean of prior
            self.mu = np.random.uniform(-1,1,self.p)
            for i in range(self.p):
                self.m0[i] = self.mu[i]

        if type == "poly":
            self.phi = lambda x,i: self.phi_poly(x,i)
            self.m0 = np.zeros(self.p)   #mean of prior
            self.mu = np.random.uniform(-1,1,self.p)
            for i in range(self.p):
                self.m0[i] = self.mu[i]


        self.I = np.eye(self.p,self.p)   #Identity matrix with dimentions pXp
        self.S0 = self.alpha**(-1)*self.I  #covariance of prior
        self.mn = self.m0    #array to hold updated mean of posterior
        self.Sn = self.S0    #matrix to hold updated variance of posterior

        self.gaussian_prior()    #Produce prior distribution

    def read_data(self,filename):

        self.x = []
        self.t = []   #Target variables

        with open(filename, "r") as infile:
            lines = infile.readlines()
            self.n = len(lines)
            for line in lines:
                words = line.split()
                self.x.append(float(words[0]))
                self.t.append(float(words[1]))

        self.x = np.array(self.x)
        self.t = np.array(self.t)
        self.create_design_matrix()

    def gaussian_prior(self):
        self.prior = sps.multivariate_normal(self.m0,self.S0)

    def create_design_matrix(self):

        self.design_matrix = np.zeros([self.n, self.p])
        self.design_matrix[:,0] = 1.0  #First comlum is 1 (bias term)

        for i in range(self.n):
            for j in range(1,self.p):
                self.design_matrix[i,j] = self.phi(self.x[i],j)

        self.design_eigvals = np.linalg.eigvals(self.design_matrix.T@self.design_matrix)

    def posterior_f(self):
        """Only valid for zero-mean isotropic Gaussian governed by a single precision parameter α"""

        self.Sn = self.alpha*self.I + self.beta*self.design_matrix.T@self.design_matrix
        self.Sn = np.linalg.inv(self.Sn)
        self.mn = self.beta*self.Sn@self.design_matrix.T@self.t
        self.posterior = sps.multivariate_normal(self.mn,self.Sn)

    def create_synt_desmat(self,gridpoints):
        n_gp = len(gridpoints)
        self.synt_des_mat = np.zeros([n_gp, self.p])
        self.synt_des_mat[:,0] = 1.0
        for i in range(n_gp):
            for j in range(1,self.p):
                self.synt_des_mat[i,j] = self.phi(gridpoints[i],j)

    def predictive_f(self,gridpoints):
        self.create_synt_desmat(gridpoints)

        predictions = self.synt_des_mat@self.mn
        cov_mat = 1/self.beta + self.synt_des_mat@self.Sn@self.synt_des_mat.T
        sig = np.sqrt(np.diag(cov_mat))

        return predictions, sig


    def y_out(self,weights,design_matrix):
        return np.sum(weights.T*design_matrix, axis=1)

    def phi_poly(self,x,i):
        return x**i

    def phi_gauss(self,x,i):
        s = 0.1
        return np.exp(-(x-self.mu[i])**2/(2*s))

    def log_marg_likelihood(self):
        self.A = np.linalg.inv(self.Sn)
        term1 = self.t - self.design_matrix@self.mn
        self.Evidence_mN = (self.beta/2)*np.linalg.norm(term1)+ (self.alpha/2)*self.mn.T@self.mn
        A_abs = np.linalg.eigvals(self.A)
        A_abs = np.prod(A_abs)

        self.marg_lik = ((self.p)/2)*np.log(self.alpha) + (self.n/2)*np.log(self.beta) - self.Evidence_mN - (1/2)*np.log(A_abs) - (self.n/2)*np.log(2*np.pi)

        return self.marg_lik

    def max_evidence(self):
        self.A = np.linalg.inv(self.Sn)
        A_eigval = np.linalg.eigvals(self.A)
        gamma = 0
        for i in range(len(A_eigval)):
            gamma += A_eigval[i]/(self.alpha + A_eigval[i])
        new_alpha = gamma/(self.mn.T@self.mn)

        sum = 0
        for i in range(self.n):
            sum +=(self.t[i]-self.mn.T@self.design_matrix[i])**2
        new_beta = 1/((1/(self.n-gamma))*sum)

        return new_alpha, new_beta

    def plot_data_space(self,gridpoints):

        std = 1
        predictions, bands = self.predictive_f(gridpoints)

        num_samples = 4
        weights = self.posterior.rvs(num_samples)  #Draws samples from posterior
        if num_samples >=2:
            for w in weights:
                plt.plot(gridpoints, self.y_out(w,self.synt_des_mat), color = "blue")
        else:
            plt.plot(gridpoints, self.y_out(weights,self.synt_des_mat), color = "blue")


        plt.scatter(self.x,self.t,marker =  'o', alpha = 0.5, edgecolor = 'k', facecolor ='None' ,label = "Real data = %i points" % len(self.x))
        plt.xlabel("x",fontsize = 16)
        plt.ylabel("y",fontsize = 16)
        plt.title("Data Space [$\\alpha =$ %.3f, $\\beta =$ %.3f]" %(self.alpha,self.beta), fontsize = 16)
        plt.title("Data Space", fontsize = 16)
        plt.xlim([-1.0,1.0])
        plt.ylim([-1.0,1.0])
        plt.plot(gridpoints, self.y_out(self.real_weights, self.synt_des_mat),"r-.", label = "Line from true weights")
        plt.legend(fontsize = 12)
        plt.savefig("./Results/Linear/Dataspace_N_"+ str(len(self.x))+".pdf")
        #plt.savefig("./Results/Linear/Hypers/Datas_a_" +str(self.alpha) +"_b_" + str(self.beta)+".pdf")
        plt.show()

        plt.scatter(self.x,self.t,marker =  'o', alpha = 0.5, edgecolor = 'k', facecolor ='None' ,label = "Real data = %i points" % len(self.x))
        plt.plot(gridpoints, predictions,color = "cyan", label = "Mean of predictive")
        plt.fill_between(gridpoints,predictions + std*bands, predictions - std*bands, facecolor='blue', alpha=0.3, zorder = 0)
        plt.plot(gridpoints, self.y_out(self.real_weights, self.synt_des_mat),"r-.", label = "Line from true weights")
        plt.xlabel("x",fontsize = 16)
        plt.ylabel("y",fontsize = 16)
        #plt.title("Predictive distribution [$\\alpha =$ %.3f, $\\beta =$ %.3f]" %(self.alpha,self.beta), fontsize = 16)
        plt.title("Predictive distribution", fontsize = 16)
        plt.xlim([-1.0,1.0])
        plt.ylim([-1.0,1.0])
        plt.legend(fontsize = 12)
        plt.savefig("./Results/Linear/Predictive_N_"+ str(len(self.x))+".pdf")
        #plt.savefig("./Results/Linear/Hypers/Pred_a_" +str(self.alpha) +"_b_" + str(self.beta)+".pdf")
        plt.show()



    def plot_sinoidal(self,gridpoints):
        std = 1

        predictions, band = self.predictive_f(gridpoints)
        self.create_synt_desmat(gridpoints)

        num_samples = 4
        self.current_weights = self.posterior.rvs(num_samples)
        if num_samples >=2:
            for w in self.current_weights:
                x, f = zip(*sorted(zip(gridpoints, self.y_out(w, self.synt_des_mat))))
                plt.plot(x,f, color = "green")
        else:
            x, f = zip(*sorted(zip(gridpoints, self.y_out(self.current_weights, self.synt_des_mat))))
            plt.plot(x, f, color = "green")

        plt.scatter(self.x,self.t, marker =  'o', alpha = 0.6, edgecolor = 'k', facecolor ='None', label = "Real data = %i points" % len(self.x))
        plt.plot(gridpoints, np.sin(2*np.pi*gridpoints) , "-.",color = 'red', label = '$\sin(2\pi x)$')
        plt.title("Data Space", fontsize = 16)
        plt.xlabel("x",fontsize = 16)
        plt.ylabel("y",fontsize = 16)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.xlim(-1.0,1.0)
        plt.legend()
        plt.savefig("./Results/GaussianBasis/Dataspace_N_"+ str(len(self.x))+".pdf")
        plt.show()

        #self.log_marg_likelihood()
        plt.fill_between(gridpoints,predictions+std*band,predictions-std*band, facecolor='blue', alpha=0.3, zorder = 0)
        plt.plot(gridpoints, predictions,color = "cyan", label = "Mean of predictive")
        plt.plot(gridpoints, np.sin(2*np.pi*gridpoints) , "-.",color = 'red', label = '$\sin(2\pi x)$')
        plt.scatter(self.x,self.t, marker =  'o', alpha = 0.6, edgecolor = 'k', facecolor ='None', label = "Real data = %i points" % len(self.x))
        plt.title("Predictive distribution", fontsize = 16)
        plt.xlabel("x",fontsize = 16)
        plt.ylabel("y",fontsize = 16)
        plt.xlim(-1.0,1.0)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.legend()
        plt.savefig("./Results/GaussianBasis/Predictive_N_"+ str(len(self.x))+".pdf")
        plt.show()


    def plot_max_evidence(self,gridpoints):
        std = 1

        predictions, band = self.predictive_f(gridpoints)


        num_samples = 1
        self.current_weights = self.posterior.rvs(num_samples)
        if num_samples >=2:
            for w in self.current_weights:
                x, f = zip(*sorted(zip(gridpoints, self.y_out(w, self.synt_des_mat))))
                plt.plot(x,f)
        else:
            x, f = zip(*sorted(zip(gridpoints, self.y_out(self.current_weights, self.synt_des_mat))))
            #plt.plot(x, f, color = "red", label = "Prediction using sampled weights")


        self.log_marg_likelihood()
        plt.fill_between(gridpoints,predictions+std*band,predictions-std*band, facecolor='blue', alpha=0.3, zorder = 0)
        plt.plot(gridpoints,predictions, color="red", label="Prediction mean")
        plt.scatter(self.x,self.t, marker =  'o', alpha = 0.5, edgecolor = 'k', facecolor ='None', label = "Real data = %i points" % len(self.x))
        plt.plot(gridpoints, np.sin(2*np.pi*gridpoints) , "-.",color = 'blue', label = 'Truth')
        plt.title("My model (L = %.4f)" %self.marg_lik, fontsize = 16)
        plt.xlabel("x",fontsize = 14)
        plt.ylabel("y",fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.legend(loc = "upper right")
        plt.xlim(-1.0,1.0)
        plt.savefig("./Results/MAXEVIDENCE_Dataspace_N_"+ str(len(self.x))+".pdf")
        plt.show()

    def plot_many_poly(self,gridpoints,ax):
        std = 1

        predictions, band = self.predictive_f(gridpoints)

        weights = self.posterior.rvs(1)
        x, f = zip(*sorted(zip(gridpoints, self.y_out(weights, self.synt_des_mat))))
        plot1 = ax.plot(x, f, label = "p = %i" % (self.p-1))

        plot2 = ax.fill_between(gridpoints,predictions + std*band,predictions - std*band, facecolor='blue', alpha=0.3, zorder = 0)
        plot3 = ax.scatter(self.x,self.t, marker =  '.', alpha = 0.5, edgecolor = 'k', facecolor ='None')
        #plot4 = ax.plot(gridpoints, np.sin(2*np.pi*gridpoints) , color = 'green', label = 'truth')
        ax.legend(fontsize = 12)
        ax.set_xlim(-1.0,1.0)
        return plot1,plot2,plot3  #,plot4


    def plot_posterior(self):

        w0 = np.linspace(-1,1,100)
        w1 = np.linspace(-1,1,100)
        w0,w1 = np.meshgrid(w0,w1)

        combo = np.dstack((w0, w1))
        combo[:, :, 0] = w0
        combo[:, :, 1] = w1

        plt.title("Posterior", fontsize = 16)
        #plt.title("Posterior [$\\alpha =$ %.3f, $\\beta =$ %.3f]" %(self.alpha,self.beta), fontsize = 16)
        plt.contourf(w0, w1, self.posterior.pdf(combo), 100)
        plt.plot(self.real_weights[0], self.real_weights[1],"w+", label= "True weighs = ["+str(self.real_weights[0]) +"," +str(self.real_weights[1]) +"]" ,markersize = 12)
        plt.legend(fontsize = 12)
        plt.xlabel('$\omega_0$', fontsize=16)
        plt.ylabel('$\omega_1$', fontsize=16)
        plt.savefig("./Results/Linear/Posterior_N_"+ str(len(self.x))+".pdf")
        #plt.savefig("./Results/Linear/Hypers/Post_a_" +str(self.alpha) +"_b_" + str(self.beta)+".pdf")
        plt.show()
