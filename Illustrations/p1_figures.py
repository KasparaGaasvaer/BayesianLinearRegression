import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

np.random.seed(5)
N = [1,3,5,20]
for n in N:
    y = np.random.normal(loc = 0, scale = 1, size = n)
    x = np.linspace(-8,8,2000)
    res = 8/1000
    density = ss.norm.pdf(x,loc = -1,scale=1.5)
    prior = density * res
    prior /= sum(prior)

    likelihood = np.zeros(len(x))
    for s in range(len(x)):
      like = 1
      for i in range(n):
        like *= ss.norm.pdf(y[i], loc = x[s], scale = 2)

      likelihood[s] = like
    likelihood /= sum(likelihood)

    posterior = likelihood * prior
    posterior /= sum(likelihood*prior)

    plt.fill_between(x,prior/res, label = "Prior", alpha =0.3, color = "lime")
    plt.fill_between(x,likelihood/res, label = "Likelihood", alpha =0.3, color = "deepskyblue")
    plt.fill_between(x, posterior/res, label = "Posterior", alpha =0.3,  color = "coral")
    plt.legend()
    plt.savefig("For_readme_PD"+"_n_" + str(n) + ".pdf")
    plt.show()
