---
title: Bayesian Linear Regression
author: lukeecust
date: 2025-06-22 15:09:00 +0800
categories: [Machine Learning, Model, Linear Regression]
tags: [LR, bayesian]
lang: en
math: true
translation_id: bayesian-linear-regression
permalink: /posts/bayesian-linear-regression/
render_with_liquid: false
---

When we talk about "linear regression," what typically comes to mind is **Ordinary Least Squares (OLS)**—that "best fit" line through data points. OLS is simple and intuitive, but it gives a single, definitive answer. It tells you: "The best slope is 2.5," but cannot answer: "How likely is the slope to be 2.4 or 2.6? How confident are we in this estimate?"

This is where **Bayesian Linear Regression** comes in. It takes us from seeking a single "best" value into a probabilistic world full of possibilities. It not only provides predictions but, more importantly, quantifies **uncertainty**.

## Core Idea: From Point Estimates to Probability Distributions

The fundamental shift in the Bayesian approach lies in how it views model parameters (e.g., weights $\mathbf{w}$).

* **Frequentist (like OLS)**: Considers parameter $\mathbf{w}$ as an unknown but **fixed** constant. Our goal is to find its best point estimate.
* **Bayesian**: Considers parameter $\mathbf{w}$ itself as a **random variable** that follows a probability distribution.

Therefore, our goal is no longer to find a single $\mathbf{w}$, but to infer the **posterior probability distribution** of $\mathbf{w}$ based on observed data. The entire process follows Bayes' theorem:

$$\begin{equation}
    p(\text{parameters} \vert \text{data}) = \frac{p(\text{data} \vert \text{parameters}) \times p(\text{parameters})}{p(\text{data})}
\end{equation}$$

Translated into our linear regression terminology:

$$\begin{equation}
p(\mathbf{w} \vert \mathcal{D}) \propto p(\mathcal{D} \vert \mathbf{w}) \times p(\mathbf{w})
\end{equation}$$

Where:
* $p(\mathbf{w} \vert \mathcal{D})$ is the **posterior probability**: Our belief about parameters $\mathbf{w}$ after seeing data $\mathcal{D}$. This is what we ultimately want to solve for.
* $p(\mathcal{D} \vert \mathbf{w})$ is the **likelihood**: Assuming the model is $y \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}, \sigma^2)$, the likelihood describes the probability of observing the current data given parameters $\mathbf{w}$.
* $p(\mathbf{w})$ is the **prior probability**: Our initial belief about parameters $\mathbf{w}$ before seeing any data. This is a major advantage of the Bayesian method as it allows us to **incorporate domain knowledge**.

So how exactly is this posterior distribution calculated? Let's dive into the mathematics.

## Mathematical Principles: Solving for the Posterior Distribution

To solve for the posterior, we need to clearly define the forms of the likelihood and prior.

### Likelihood Function $p(\mathcal{D} \vert \mathbf{w})$

We assume there is Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ between the model output $y$ and the true value $\mathbf{w}^T \mathbf{x}$. For mathematical convenience, we often use **precision** $\beta = 1/\sigma^2$ to represent this. Therefore, for a single data point, the probability is:
$$\begin{equation}
p(y_i \vert \mathbf{x_i}, \mathbf{w}, \beta) = \mathcal{N}(y_i \vert \mathbf{w}^T \mathbf{x_i}, \beta^{-1})
\end{equation}$$

For the entire i.i.d. dataset $\mathcal{D} = \{(\mathbf{X}, \mathbf{y})\}$, the likelihood function is the product of probabilities for all data points:
$$\begin{equation}
p(\mathbf{y} \vert \mathbf{X}, \mathbf{w}, \beta) = \prod_{i=1}^{N} \mathcal{N}(y_i \vert \mathbf{w}^T \mathbf{x_i}, \beta^{-1})
\end{equation}$$

### Prior Distribution $p(\mathbf{w})$

To make computation feasible, we choose a **conjugate prior**. When the prior and likelihood are conjugate, their product (the posterior) will have the same functional form as the prior. For Gaussian likelihood, its conjugate prior is also a Gaussian distribution. We assume $\mathbf{w}$ follows a multivariate Gaussian distribution with mean $\mathbf{m}_0$ and covariance $\mathbf{S}_0$:
$$\begin{equation}
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \vert \mathbf{m}_0, \mathbf{S}_0)
\end{equation}$$

Typically, we choose a simple zero-mean prior, $\mathbf{m}_0 = \mathbf{0}$, with covariance matrix $\mathbf{S}_0 = \alpha^{-1} \mathbf{I}$, where $\alpha$ is a hyperparameter representing the precision of our prior belief about the magnitude of weights $\mathbf{w}$.

### Deriving the Posterior Distribution $p(\mathbf{w} \vert \mathcal{D})$

Now, we multiply the likelihood and prior. To simplify, we work with their logarithmic forms and ignore constant terms that don't involve $\mathbf{w}$:
$$\begin{equation}
\begin{aligned}
\ln p(\mathbf{w} \vert \mathcal{D}) &\propto \ln p(\mathbf{y} \vert \mathbf{X}, \mathbf{w}, \beta) + \ln p(\mathbf{w}) \\
&= \ln \left[ \exp(-\frac{\beta}{2} (\mathbf{y} - \mathbf{Xw})^T (\mathbf{y} - \mathbf{Xw})) \right] + \ln \left[ \exp(-\frac{1}{2} (\mathbf{w} - \mathbf{m}_0)^T \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0)) \right] \\
&= -\frac{\beta}{2} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{X}^T\mathbf{Xw}) - \frac{1}{2} (\mathbf{w}^T\mathbf{S}_0^{-1}\mathbf{w} - 2\mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{w} + \mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{m}_0) + \text{const}
\end{aligned}
\end{equation}$$

Our goal is to arrange this expression into the standard form of a logarithm of a Gaussian distribution with respect to $\mathbf{w}$: $-\frac{1}{2}(\mathbf{w}-\mathbf{m}_N)^T \mathbf{S}_N^{-1} (\mathbf{w}-\mathbf{m}_N)$. We focus only on terms containing $\mathbf{w}$:
* **Quadratic terms in $\mathbf{w}$**: $-\frac{1}{2}(\beta \mathbf{w}^T\mathbf{X}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{S}_0^{-1}\mathbf{w}) = -\frac{1}{2}\mathbf{w}^T(\beta \mathbf{X}^T\mathbf{X} + \mathbf{S}_0^{-1})\mathbf{w}$
* **Linear terms in $\mathbf{w}$**: $\beta \mathbf{y}^T\mathbf{Xw} + \mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{w} = (\beta \mathbf{y}^T\mathbf{X} + \mathbf{m}_0^T\mathbf{S}_0^{-1})\mathbf{w}$

By comparing with the quadratic term $-\frac{1}{2}\mathbf{w}^T\mathbf{S}_N^{-1}\mathbf{w}$ and linear term $\mathbf{m}_N^T\mathbf{S}_N^{-1}\mathbf{w}$ from the standard Gaussian logarithmic form when expanded, we get:
$$\begin{equation}
\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \mathbf{X}^T \mathbf{X}
\end{equation}$$
$$\begin{equation}
\mathbf{m}_N^T\mathbf{S}_N^{-1} = \mathbf{y}^T(\beta\mathbf{X}^T) + \mathbf{m}_0^T\mathbf{S}_0^{-1}
\end{equation}$$

From the first equation, we get the posterior covariance $\mathbf{S}_N$:
$$\begin{equation}
\mathbf{S}_N = (\mathbf{S}_0^{-1} + \beta \mathbf{X}^T \mathbf{X})^{-1}
\end{equation}$$

Substituting $\mathbf{S}_N^{-1}$ into the second equation and solving for $\mathbf{m}_N$, we get the posterior mean $\mathbf{m}_N$:
$$\begin{equation}
\mathbf{m}_N = \mathbf{S}_N (\mathbf{S}_0^{-1} \mathbf{m}_0 + \beta \mathbf{X}^T \mathbf{y})
\end{equation}$$

Through mathematical derivation, we've obtained a new Gaussian distribution $\mathcal{N}(\mathbf{w} \vert \mathbf{m}_N, \mathbf{S}_N)$. This is our final belief about parameters $\mathbf{w}$ after updating with data!

### Predicting New Data Points

When we have a new data point $$\mathbf{x}_{\ast}$$, the distribution of the predicted value $$y_{\ast}$$ can be obtained by integrating over all possible $$\mathbf{w}$$:
$$\begin{equation}
p(y_{\ast} \vert \mathbf{x}_{\ast}, \mathcal{D}) = \int p(y_{\ast} \vert \mathbf{x}_{\ast}, \mathbf{w}) p(\mathbf{w} \vert \mathcal{D}) d\mathbf{w}
\end{equation}$$

The result of this integral is also a Gaussian distribution with mean $$\mathbf{m}_N^T \mathbf{x}_{\ast}$$ and variance:
$$\begin{equation}
sigma_{pred}^2 = \underbrace{\frac{1}{\beta}}_{\text{Inherent data noise}} + \underbrace{\mathbf{x}_*^T \mathbf{S}_N \mathbf{x}_*}_{\text{Model parameter uncertainty}}
\end{equation}$$

This prediction variance perfectly illustrates the essence of the Bayesian approach: **Total uncertainty = Randomness inherent in the data + Uncertainty in our knowledge of the model.**

## Two Implementation Paths: Analytical Solution vs. Sampling

The mathematical derivation above provides us with an **analytical solution**. As long as we choose conjugate priors and likelihoods, we can directly compute the posterior distribution.
1. **Analytical/Approximate Methods**:
   These methods are efficient and fast. Scikit-learn's `BayesianRidge` is based on this idea, finding approximate solutions to the posterior distribution through optimization.

2. **Markov Chain Monte Carlo (MCMC) Sampling**:
   But what if we want to use more complex priors (like a bimodal distribution), or if the model's likelihood is not Gaussian? Analytical solutions no longer exist. In these cases, we need sampling techniques like MCMC to draw thousands of samples from the posterior distribution that is difficult to compute directly. `PyMC` is a powerful tool created for this purpose.

Now, let's see how these two paths are implemented in code.

## Code Implementation

### Scikit-learn's `BayesianRidge` 

`BayesianRidge` applies the principles we derived above and uses **empirical Bayes** to automatically estimate hyperparameters $\alpha$ and $\beta$. It's very suitable as a direct replacement for standard linear regression.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 1. Generate simulated data
def generate_data(n_samples=30, noise_std=0.5):
    np.random.seed(42)
    X = np.linspace(-5, 5, n_samples)
    # True parameters
    true_w = 0.5
    true_b = -1
    y = true_w * X + true_b + np.random.normal(0, noise_std, size=n_samples)
    return X.reshape(-1, 1), y

# Generate data
X_train, y_train = generate_data()

# 2. Create and train the model
# BayesianRidge automatically estimates alpha (weight precision) and lambda (noise precision)
br = BayesianRidge(compute_score=True)
br.fit(X_train, y_train)

print(f"Scikit-learn BayesianRidge:")
print(f"Estimated weights (w): {br.coef_[0]:.4f}")
print(f"Estimated intercept (b): {br.intercept_:.4f}")
print(f"Estimated alpha (precision of weights): {br.alpha_:.4f}")
print(f"Estimated lambda (precision of noise): {br.lambda_:.4f}")

# 3. Create test points and predict
X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
y_mean_sk, y_std_sk = br.predict(X_test, return_std=True)

# 4. Visualize results
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.7, zorder=2)
plt.plot(X_test, y_mean_sk, label="BayesianRidge Mean Prediction", color="purple", zorder=3)
plt.fill_between(X_test.ravel(), y_mean_sk - y_std_sk, y_mean_sk + y_std_sk,
                 color="purple", alpha=0.2, label="Uncertainty (±1 std)", zorder=1)

plt.title("BayesianRidge (Scikit-learn)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

**Result Analysis**: `BayesianRidge` quickly provides a prediction with uncertainty intervals. The width of this uncertainty interval (purple shade) is mathematically based on the prediction variance formula we just derived.

**Main Parameter Explanations:**
- `n_iter`: Maximum number of iterations, default is 300
- `tol`: Convergence threshold, default is 1e-3
- `alpha_1`, `alpha_2`: Gamma prior parameters for $\alpha$ (weight precision), defaults are 1e-6 and 1e-6
- `lambda_1`, `lambda_2`: Gamma prior parameters for $\beta$ (noise precision), defaults are 1e-6 and 1e-6
- `compute_score`: Whether to compute log marginal likelihood at each iteration, default is False
- `fit_intercept`: Whether to calculate the intercept, default is True
- `normalize`: Deprecated, use `StandardScaler` instead
- `copy_X`: Whether to copy X, default is True

**Prediction Method Parameters:**
- `return_std`: If True, returns the standard deviation of the prediction, default is False
- `return_cov`: If True, returns the covariance of the prediction, default is False

### PyMC

`PyMC` is completely different. We don't care if an analytical solution exists; instead, we directly **describe our probabilistic model** to the computer and let the MCMC sampler explore the posterior distribution.

**Main Functions and Distributions Explanation:**

```python
import pymc as pm
import arviz as az

# Use the same data as above
# X_train, y_train

# 1. Define PyMC model
with pm.Model() as bayesian_linear_model:
    # Define prior distributions
    # Use weakly informative normal prior for intercept b
    b = pm.Normal('intercept', mu=0, sigma=10)
    # Use weakly informative normal prior for slope w
    w = pm.Normal('slope', mu=0, sigma=10)
    # Use weakly informative half-normal prior for data noise standard deviation sigma (must be positive)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # Define linear model (likelihood mean)
    mu = w * X_train.ravel() + b

    # Define likelihood distribution
    # y_obs is our observed data
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)

    # 2. Run MCMC sampler
    # Sample 2000 times, warm-up/tune 1000 times
    trace = pm.sample(2000, tune=1000, cores=1)
    
# 3. Analyze and visualize results
print("\nPyMC Model Summary:")
az.summary(trace, var_names=['intercept', 'slope', 'sigma'])

# 4. Generate posterior predictions
with bayesian_linear_model:
    # Generate posterior prediction samples at test points
    post_pred = pm.sample_posterior_predictive(trace, var_names=['y_obs'], samples=1000)

# Extract prediction results
# PyMC 4.0+ uses .stack(sample=("chain", "draw"))
y_pred_samples = post_pred.posterior_predictive['y_obs'].stack(sample=("chain", "draw")).values.T
# Replace with X_test
mu_test = trace.posterior['slope'].values * X_test.ravel() + trace.posterior['intercept'].values
y_mean_pm = mu_test.mean(axis=1)
# Use ArviZ to calculate HDI (Highest Density Interval), a more robust Bayesian uncertainty measure than standard deviation
hdi_data = az.hdi(mu_test.T, hdi_prob=0.94) # 94% HDI ~ 2 std

# 5. Visualization
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.7, zorder=2)
plt.plot(X_test, y_mean_pm, label="PyMC Mean Prediction", color="green", zorder=3)
plt.fill_between(X_test.ravel(), hdi_data[:,0], hdi_data[:,1],
                 color="green", alpha=0.2, label="Uncertainty (94% HDI)", zorder=1)

# Draw a few lines sampled from the posterior
for i in np.random.randint(0, mu_test.shape[1], 10):
    plt.plot(X_test, mu_test[:, i], color='gray', alpha=0.3, lw=1)


plt.title("Full Bayesian Regression (PyMC)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

**Result Analysis**: The result from `PyMC` has more "Bayesian flavor." What we get is not a single set of posterior distribution parameters, but thousands of samples drawn from the posterior distribution (the `trace` object). These samples form an empirical approximation of the entire posterior distribution. 

**PyMC Model Building Functions:**
- `pm.Model()`: Creates a Bayesian model container
- `pm.Normal(name, mu, sigma)`: Creates a normal distribution variable, `mu` is the mean, `sigma` is the standard deviation
- `pm.HalfNormal(name, sigma)`: Creates a half-normal distribution variable, used for non-negative parameters
- `pm.sample(draws, tune, cores)`:
  - `draws`: Number of samples
  - `tune`: Number of warm-up iterations for adjusting the sampler
  - `cores`: Number of cores for parallel computation

**ArviZ Result Analysis Functions:**
- `az.summary(trace, var_names)`: Returns statistical summary of posterior distribution
- `az.hdi(data, hdi_prob)`: Calculates highest density interval, `hdi_prob` specifies probability mass (e.g., 0.94 for 94%)

### `BayesianRidge` vs. `PyMC`

| Feature | `sklearn.BayesianRidge` | `PyMC` |
| :--- | :--- | :--- |
| **Mathematical Basis** | **Relies on analytical solution** (or its approximation) | **Does not rely on analytical solution**, uses MCMC sampling |
| **Methodology** | Empirical Bayes | Fully Bayesian |
| **Flexibility** | Low: Fixed model structure (Gaussian prior and likelihood) | **Very high**: Can customize any prior, likelihood, and complex model structure |
| **Speed** | **Very fast** | Slower, depends on model complexity and data size |
| **Ease of Use** | **Very simple**, API consistent with other sklearn models | Steeper learning curve, requires understanding of Bayesian modeling concepts |
| **Output Information** | Posterior mean and covariance matrix of parameters | **Complete posterior distribution samples for all parameters** |
| **Suitable Scenarios** | Quickly adding uncertainty estimates to standard linear problems | Complex/non-standard models, in-depth parameter uncertainty analysis, hierarchical models, etc. |

**How to Choose?**

* **Choose `BayesianRidge`**: When your problem fits the basic assumptions of linear regression (Gaussian noise), and you're satisfied with the regularization effect from conjugate priors. It's an efficient, practical choice, backed by the elegant mathematics we just derived.
* **Choose `PyMC`**: When you're facing a complex model that can't be solved with simple mathematical formulas, or when you want complete control over every aspect of the model (priors, likelihood). It liberates us from being limited to models with analytical solutions, allowing us to explore a broader world of Bayesian modeling.

## Summary

Bayesian linear regression shifts us from seeking a single "best answer" to embracing and quantifying uncertainty. Understanding the mathematical principles behind it, we can see that `BayesianRidge` is an elegant application of this theory, while `PyMC` provides a powerful and general path when the theory cannot be directly solved.

I hope this mathematically detailed version helps you understand Bayesian linear regression more thoroughly, and enables you to comfortably choose between the convenience of Scikit-learn and the power of PyMC!
