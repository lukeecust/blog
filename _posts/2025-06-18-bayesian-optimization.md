---
title: Bayesian Optimization - From Mathematical Principles to Hyperparameter Tuning
author: lukeecust
date: 2025-06-18 23:09:00 +0800
categories: [Machine Learning, Hyperparameter Optimization]
tags: [optimization, bayesian, HPO]
lang: en
math: true
translation_id: bayesian-optimization
permalink: /posts/bayesian-optimization/
render_with_liquid: false
---

In machine learning, model performance largely depends on the selection of hyperparameters. However, finding the optimal hyperparameter combination—hyperparameter optimization (HPO)—remains a challenging task. This article focuses on one of the mainstream methods in this field: Bayesian Optimization (BO). We will start from its mathematical principles, analyze the core components of Bayesian optimization in depth, and demonstrate its effectiveness in practical applications using the `hyperopt` library.

## Hyperparameter Optimization: A Black-box Optimization Problem

Machine learning model hyperparameters must be set before training begins. The relationship between hyperparameters and model performance is typically:

*   **Black-box**: We cannot write an explicit mathematical function $f(x)$ for model performance (e.g., validation accuracy) in terms of hyperparameters. We can only obtain a performance score $f(x)$ for a specific hyperparameter combination $x$ through a complete training and evaluation process, without access to gradient information.
*   **Expensive to evaluate**: Each evaluation of $f(x)$ may take hours or even days.
*   **High-dimensional and complex**: The search space has high dimensionality and may include continuous, discrete, and conditional parameters.

Traditional grid search and random search are inefficient because they fail to effectively utilize historical information. Bayesian optimization, as a **Sequential Model-Based Optimization (SMBO)** strategy, constructs a probabilistic model of the objective function and intelligently selects the next evaluation point, greatly improving optimization efficiency.

## Mathematical Framework of Bayesian Optimization

The goal of Bayesian optimization is to solve for the global extremum of a black-box function $f(x)$ (maximization, for example):

$$
\begin{equation}
x^* = \underset{x \in \mathcal{X}}{\text{argmax}} \, f(x)  
\end{equation}
$$

where $\mathcal{X}$ is the search space of hyperparameters. The algorithm consists of two key components:

1.  **Probabilistic Surrogate Model**: A simple model that approximates the true black-box function $f(x)$, capable of predicting the function value at any point and quantifying its **uncertainty**.
2.  **Acquisition Function**: Based on the surrogate model's predictions, a function is constructed to evaluate the "value" of sampling at each candidate point for the next iteration, and its maximum is found as the next sampling point.

The iterative process of the algorithm is as follows:
1.  **Initialization**: Randomly sample a few points to form the initial dataset $D_0 = \{(x_1, y_1), ..., (x_n, y_n)\}$.
2.  **Iterative loop** for $t=1, 2, ...$:
3. 
  a.  Use current data $$D_{t-1}$$ to fit the probabilistic surrogate model, updating the posterior probability distribution of $$f(x)$$.

  b.  Optimize the acquisition function $$\alpha(x)$$ to find the next evaluation point: $$x_t = \underset{x \in \mathcal{X}}{\text{argmax}} \, \alpha(x)$$.

  c.  Evaluate $$y_t = f(x_t)$$.

  d.  Update the dataset $$D_t = D_{t-1} \cup \{(x_t, y_t)\}$$.
4.  After **termination**, return the best point from all evaluated points.

###  Component One: Probabilistic Surrogate Model - Learning the Objective Function

The surrogate model is the core of Bayesian optimization. Based on observed points, it provides a probability distribution of the objective function values across the entire search space. The two most mainstream methods are Gaussian Process Regression (GPR) and Tree-structured Parzen Estimator (TPE).

####  Gaussian Process Regression (GPR): Classic Theoretical Foundation

A **Gaussian Process (GP)** assumes that the joint distribution of function values for any set of input points follows a multivariate Gaussian distribution. A GP is defined by its **mean function** $m(x)$ and **covariance function (kernel function)** $k(x, x')$:
$$
\begin{equation}
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
\end{equation}
$$
**Gaussian Process Regression (GPR)** uses the GP prior and observed data $D_t$ to derive the posterior distribution at any new point $$x_\ast$$, which remains a Gaussian distribution $$p(f(x_\ast) | D_t, x_\ast) = \mathcal{N}(\mu_t(x_\ast), \sigma_t^2(x_\ast))$$, with analytical solutions for mean and variance:
$$
\begin{equation}
\mu_t(x_\ast) = \mathbf{k}_\ast^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{y}
\end{equation}
$$
$$
\begin{equation}
\sigma_t^2(x_\ast) = k(x_\ast, x_\ast) - \mathbf{k}_\ast^T (\mathbf{K} + \sigma_n^2\mathbf{I})^{-1} \mathbf{k}_\ast
\end{equation}
$$
where $$\mathbf{K}$$ is the kernel matrix, $$\mathbf{k}_\ast$$ is the kernel vector between the new point and observed points, and $$\mathbf{y}$$ is the vector of observed values. $$\mu_t(x_\ast)$$ is the best prediction for $$f(x_\ast)$$, while $$\sigma_t^2(x_\ast)$$ quantifies the uncertainty of this prediction.

####  Tree-structured Parzen Estimator (TPE): Practical and Efficient Alternative

Unlike GPR which directly models $$p(y\vert x)$$, **TPE (Tree-structured Parzen Estimator)** uses Bayes' theorem to model $$p(x\vert y)$$ and $$p(y)$$ instead. Its core idea is:

1.  **Data partitioning**: Based on a quantile threshold $$y^\ast$$ (e.g., the best 15% of all observed values), historical observation data is divided into a "good" set $$\mathcal{D}_g = \{(x,y) \vert y < y^\ast\}$$ and a "bad" set $$\mathcal{D}_b = \{(x,y) \vert y \ge y^\ast\}$$.
2.  **Density modeling**: Establish probability density models for hyperparameters $$x$$ for these two sets of data. The distribution of good parameters is $$l(x) = p(x\vert y<y^\ast)$$, and the distribution of bad parameters is $$g(x) = p(x\vert y \ge y^\ast)$$. These density functions are typically estimated using Parzen windows (i.e., kernel density estimation).
3.  **Acquisition function optimization**: TPE's acquisition function is related to Expected Improvement (EI), with the ultimate goal of finding points $$x$$ that maximize the ratio $$l(x)/g(x)$$. Intuitively, this means we are looking for parameter points that have high probability in the "good" distribution but low probability in the "bad" distribution.

The main advantage of TPE is its natural ability to handle complex tree-structured search spaces with conditional and discrete variables, and its greater computational scalability compared to GPR.

| Feature | Gaussian Process Regression (GPR) | Tree-structured Parzen Estimator (TPE) |
| :--- | :--- | :--- |
| **Core Idea** | Models $p(y\vert x)$, directly estimating the distribution of function values | Models $p(x\vert y)$, estimating parameter distributions in good/bad cases |
| **Mathematical Foundation** | Gaussian processes, Bayesian linear regression | Bayes' theorem, kernel density estimation (Parzen window) |
| **Parameter Space** | Best for **continuous** and **low-dimensional** spaces | Excellent for handling **discrete, conditional parameters** and **high-dimensional** spaces |
| **Computational Complexity** | $O(n^3)$, limited by kernel matrix inversion, poor scalability | $O(n \log n)$, better scalability |
| **Parallelism** | Difficult to parallelize, inherently sequential | Easier to parallelize, can generate multiple candidate points by sampling from $l(x)$ |
| **Common Tools** | `scikit-optimize`, `GPyOpt` | `hyperopt`, `Optuna` |

### Component Two: Acquisition Function - Balancing Exploration and Exploitation

The acquisition function $\alpha(x)$ uses the surrogate model's predictions $(\mu(x), \sigma(x))$ to decide the next sampling point, cleverly balancing **exploitation** and **exploration**. Let $f^+ = \max_{i} f(x_i)$ be the current best observed value.

1.  **Probability of Improvement (PI)**:
  $$
  \begin{equation}
  \alpha_{PI}(x) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)
  \end{equation}
  $$
  where $\Phi(\cdot)$ is the standard normal CDF, and $\xi$ is a tuning factor.

2.  **Expected Improvement (EI)**: The most commonly used acquisition function, calculating the expected improvement.
  $$\begin{equation}
  \alpha_{EI}(x) = (\mu(x) - f^+) \Phi(Z) + \sigma(x)\phi(Z) \quad \text{where} \quad Z = \frac{\mu(x) - f^+}{\sigma(x)}
  \end{equation}$$
  where $\phi(\cdot)$ is the standard normal PDF.

3.  **Upper Confidence Bound (UCB)**:
  $$\begin{equation}
  \alpha_{UCB}(x) = \mu(x) + \kappa \sigma(x)
  \end{equation}$$
  where $\kappa \ge 0$ controls the weight of exploration.

## AutoML, HPO, NAS

When discussing Bayesian optimization, it's necessary to clarify its position in the Automated Machine Learning (AutoML) ecosystem:

*   **AutoML**: A grand goal aiming to automate the entire machine learning pipeline, including data preprocessing, feature engineering, model selection, hyperparameter optimization (HPO), and neural architecture search (NAS).
*   **HPO vs. NAS**: Both fall under the umbrella of AutoML, but optimize different objects.

| Item | HPO (Hyperparameter Optimization) | NAS (Neural Architecture Search) |
| :--- | :--- | :--- |
| **Optimization Object** | Model hyperparameters (e.g., learning rate, batch size, regularization coefficients) | Model network structure (e.g., number of layers, convolution kernel size, connection methods) |
| **Search Space** | Usually combinations of scalar/continuous/discrete values | Usually graph structures, module combinations, and other discrete and complex spaces |
| **Optimization Methods** | Grid/random search, **Bayesian optimization**, evolutionary algorithms, etc. | Reinforcement learning, evolutionary algorithms, **Bayesian optimization**, gradient methods, etc. |

Bayesian optimization is an advanced and efficient method for implementing HPO, and is sometimes also applied to NAS tasks.

## Using `hyperopt` (TPE) for XGBoost Tuning

`hyperopt` is a popular Python library implementing the SMBO framework, with TPE as its core algorithm. Below, we use it to find optimal hyperparameters for an `XGBoost` classifier.

### Step 1: Define the Objective Function $f(x)$
```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Define objective function
def objective(params):
  # hyperopt passes floats, some parameters need to be converted to integers
  params['max_depth'] = int(params['max_depth'])
  params['n_estimators'] = int(params['n_estimators'])
  
  clf = xgb.XGBClassifier(
    **params,
    use_label_encoder=False,
    eval_metric='mlogloss'
  )
  
  # Evaluate the model using cross-validation, return negative accuracy as loss
  accuracy = cross_val_score(clf, X, y, cv=5).mean()
  loss = 1 - accuracy
  
  # hyperopt requires a dictionary return, must include 'loss' and 'status'
  return {'loss': loss, 'status': STATUS_OK, 'accuracy': accuracy}
```

### Step 2: Define the Search Space $\mathcal{X}$
```python
# 3. Define search space
space = {
  'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
  'max_depth': hp.quniform('max_depth', 3, 15, 1),
  'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
  'gamma': hp.uniform('gamma', 0, 0.5),
  'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
  'subsample': hp.uniform('subsample', 0.5, 1.0)
}
```
**Parameter Descriptions:**

- **`hp.quniform(label, low, high, q)`**: Discrete uniform distribution
  - `label`: Parameter label name
  - `low`: Minimum value
  - `high`: Maximum value
  - `q`: Discretization step size

- **`hp.uniform(label, low, high)`**: Continuous uniform distribution
  - `label`: Parameter label name
  - `low`: Minimum value
  - `high`: Maximum value

- **`hp.loguniform(label, low, high)`**: Log-uniform distribution
  - `label`: Parameter label name
  - `low`: Logarithm of minimum value
  - `high`: Logarithm of maximum value

- **Other available distribution functions:**
  - `hp.choice(label, options)`: Choose from options
  - `hp.randint(label, upper)`: Random integer [0, upper)
  - `hp.normal(label, mu, sigma)`: Normal distribution
  - `hp.lognormal(label, mu, sigma)`: Log-normal distribution
  
### Step 3: Run Bayesian Optimization
```python
# 4. Run optimization
trials = Trials()

best_params = fmin(
  fn=objective,
  space=space,
  algo=tpe.suggest,  # Explicitly specify using the TPE algorithm
  max_evals=100,
  trials=trials
)

print("\n" + "="*50)
print("Optimization completed")
print("="*50)
# fmin returns parameters that minimize loss, but some values may be floats and need to be adjusted
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
print("Best hyperparameter combination found:")
print(best_params)

# Get the best trial result from the trials object
best_trial = trials.best_trial
print(f"\nBest loss value (1 - accuracy): {best_trial['result']['loss']:.4f}")
print(f"Corresponding cross-validation accuracy: {best_trial['result']['accuracy']:.4f}")
```


**Parameter Descriptions:**

- **`fmin()`** parameters:
  - `fn`: Objective function (required)
  - `space`: Search space (required)
  - `algo`: Optimization algorithm (required, e.g., `tpe.suggest`)
  - `max_evals`: Maximum number of evaluations (required)
  - `trials`: Trials object (optional, defaults to None)
  - `rstate`: Random state (optional, defaults to None)
  - `verbose`: Verbose output (optional, defaults to 0)
  - `return_argmin`: Whether to return the minimum value parameters (optional, defaults to True)
  - `show_progressbar`: Whether to display a progress bar (optional, defaults to True)

- **`Trials()`** parameters:
  - `exp_key`: Experiment key (optional, defaults to None)
  - `refresh`: Whether to refresh (optional, defaults to True)

- **Other available algorithms:**
  - `tpe.suggest`: TPE algorithm (Tree-structured Parzen Estimator)
  - `rand.suggest`: Random search
  - `anneal.suggest`: Simulated annealing

## Conclusion

Bayesian optimization provides a powerful and rigorous mathematical framework for solving high-cost black-box optimization problems. It constructs a surrogate model (such as GPR or TPE) to approximate the true objective function and uses an acquisition function to intelligently balance exploration and exploitation, thereby efficiently finding the optimal solution. GPR provides a solid theoretical foundation for Bayesian optimization, while TPE demonstrates excellent performance and scalability in handling complex, high-dimensional practical problems.
