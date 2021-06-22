# Bayesian Linear Regression
Repository containing the work of Kaspara Skovli Gåsvær on Bayesian linear regression. Much of the work found in this repository is based on the books *Pattern Recognition and Machine Learning* ([Bishop, 2006](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)) and *Data Analysis: A Bayesian Tutorial* ([Sivia, 2006](http://aprsa.villanova.edu/files/sivia.pdf)).


<img src="https://github.com/KasparaGaasvaer/BayesianLinearRegression/blob/main/Illustrations/readmeimg.png" alt="Illustration of prior, likelihood and posterior distribution." width="192" height="144">

### Articles
This repository contains two papers written on the subject. *An Introduction to Bayesian Linear Regression* ([Gåsvær, 2021](https://github.com/KasparaGaasvaer/BayesianLinearRegression/blob/main/Articles/An_Introduction_to_Bayesian_Linear_Regression.pdf)) covers theoretical background and serves as an easy introduction to the subject intended for those with background mainly from other natural sciences than statistics. *Practical Implementation of Bayesian Inference on Linear Regression Problems* ([Gåsvær, 2021](https://github.com/KasparaGaasvaer/BayesianLinearRegression/blob/main/Articles/Practical_Implementation_of_Bayesian_Inference_on_Linear_Regression_Problems.pdf)) makes use of the theory presented in the former paper for practical implementation purposes. It looks at the implications of model selection and parameter estimation as well as how to interpret the resulting parameters in terms of priors and posteriors.

### Code
Included is a `main.py` file which can be run to reproduce the data in the articles. The script makes use of relevant methods from the `Bayesian_Linear_Regression` class. N is set to 100 data points which can easily be changed at the top of the main program along with the values of precision parameters and the coefficients of the polynomial used to create data as well as the standard deviation of said data. To run the main file type the following into your terminal

```console
python3 main.py task
```  

Where the valid choices of "task" consists of:
-  `linear_fit` : fits data from polynomial function using a linear polynomial basis function for increasing amount of N accessible data points.

- `gridsearch_hyperparams` : Performs a grid search on sets of precision parameters fitting the same data as in linear_fit.

- `sinusoidal_data_gauss_basis` : fits data from a sinusoidal function using 9 gaussian basis functions for increasing amount of N accessible data points.

- `sinusoidal_data_poly_basis` : fits data from a sinusoidal function using polynomial basis functions of degree 0-9. Calculates log marginal likelihood score for each degree of polynomial.

- `gaussian_data_poly_basis` : fits data from a polynomial of degree 8 using polynomial basis functions of degree 0-9. Calculates log marginal likelihood score for each degree of polynomial.

- `skl_compare` : maximizes the evidence function and updates precision parameters iteratively. Fits data from sinusoidal function using optimized parameters. Compares weights to those generated from Scikit-Learn BayisianRidge model used on the same data set.
