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

## Using `optuna` (TPE) for XGBoost Tuning

`optuna` is a modern automated hyperparameter optimization framework that uses the TPE algorithm by default and provides a very intuitive and flexible "Define-by-Run" API. Below, we'll use it to find the optimal hyperparameters for an `XGBoost` classifier.

### Step 1: Load the Data
First, we need to prepare the data for model training and evaluation.

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import optuna

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target
```

### Step 2: Define the Objective Function $f(x)$
This is where the core difference between `optuna` and `hyperopt` lies. In `optuna`, the search space is dynamically defined within the objective function through the `trial` object. The objective function takes a `trial` object as a parameter and returns a value to be optimized (such as loss or accuracy).

```python
# 2. Define the objective function
def objective(trial):
  # Inside the function, suggest hyperparameter values through the trial object
  # This is the "Define-by-Run" API
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
    'max_depth': trial.suggest_int('max_depth', 3, 15),
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
    'gamma': trial.suggest_float('gamma', 0, 0.5),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    
    # Fixed parameters can be written directly
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
  }
  
  clf = xgb.XGBClassifier(**params)
  
  # Use cross-validation to evaluate the model, return negative accuracy as loss
  accuracy = cross_val_score(clf, X, y, cv=5).mean()
  
  # Optuna will handle this return value according to the study's optimization direction
  # We want to minimize the loss, so return 1 - accuracy
  loss = 1 - accuracy
  return loss
```

**`trial` Object Method Explanation:**
The `trial` object is the core tool for defining the search space within the objective function.

- **`trial.suggest_int(name, low, high, step=1, log=False)`**: Suggests an integer.
  - `name` (str): The parameter name, must be unique within a `study`.
  - `low` (int): The lower bound of the search range (inclusive).
  - `high` (int): The upper bound of the search range (inclusive).
  - `step` (int): The step size for suggested values. For example, `step=25` will select from `[50, 75, 100, ...]`.
  - `log` (bool): If `True`, sampling is performed on a logarithmic scale, suitable for integers across orders of magnitude.

- **`trial.suggest_float(name, low, high, step=None, log=False)`**: Suggests a floating-point number.
  - `name` (str): The parameter name.
  - `low` (float): The lower bound of the search range (inclusive).
  - `high` (float): The upper bound of the search range (inclusive).
  - `step` (float, optional): If specified, suggests discrete floating-point values.
  - `log` (bool): If `True`, sampling is performed on a logarithmic scale. This is very effective for parameters such as `learning_rate`, as it allows more uniform exploration of the regions between `0.001` and `0.01`, as well as between `0.01` and `0.1`.

- **`trial.suggest_categorical(name, choices)`**: Suggests a category from a list.
  - `name` (str): The parameter name.
  - `choices` (list): A list containing all possible options, e.g., `['gbtree', 'dart']`.

- **`trial.suggest_discrete_uniform(name, low, high, q)`**: Suggests a value from a discrete uniform distribution.
  - `name` (str): The parameter name.
  - `low` (float): The lower bound of the search range (inclusive).
  - `high` (float): The upper bound of the search range (inclusive).
  - `q` (float): The discretization step.

- **`trial.suggest_loguniform(name, low, high)`**: Suggests a value from a log-uniform distribution (deprecated, recommend using `suggest_float` with `log=True`).

- **`trial.suggest_uniform(name, low, high)`**: Suggests a value from a uniform distribution (deprecated, recommend using `suggest_float`).

**Mapping Relationship with `hyperopt` Distribution Functions:**

| hyperopt | optuna | Meaning |
|----------|--------|------|
| `hp.choice(label, options)` | `trial.suggest_categorical(name, choices)` | Choose a value from discrete options, suitable for categorical parameters |
| `hp.randint(label, upper)` | `trial.suggest_int(name, 0, upper-1)` | Return a random integer in the range [0, upper-1] |
| `hp.uniform(label, low, high)` | `trial.suggest_float(name, low, high)` | Sample floating-point numbers uniformly in the range [low, high] |
| `hp.quniform(label, low, high, q)` | `trial.suggest_float(name, low, high, step=q)` | Sample discrete values uniformly in the range [low, high] with step size q |
| `hp.loguniform(label, low, high)` | `trial.suggest_float(name, np.exp(low), np.exp(high), log=True)` | Sample uniformly in log-space, suitable for parameters that need to be explored across multiple orders of magnitude |
| `hp.qloguniform(label, low, high, q)` | `trial.suggest_float(name, np.exp(low), np.exp(high), log=True, step=q)` | Sample discrete values uniformly in log-space with step size q |
| `hp.normal(label, mu, sigma)` | *No direct equivalent, can be implemented through custom samplers* | Sample from a normal distribution (mean mu, standard deviation sigma) |
| `hp.lognormal(label, mu, sigma)` | *No direct equivalent, can be implemented through custom samplers* | Sample from a log-normal distribution, suitable for non-negative parameters with a long-tail distribution |

### Step 3: Create a Study and Run Optimization
In `optuna`, we first create a `study` object to manage the entire optimization process, then call its `optimize` method to start the optimization.

```python
# 3. Create a study object and run optimization
# direction='minimize' indicates that our goal is to minimize the return value of the objective function
study = optuna.create_study(direction='minimize')

# Call the optimize method to start optimization
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

**Function Explanation:**

- **`optuna.create_study()`**: Creates a `study` object, which is the controller for the optimization task.
  - `direction` (str): Optimization direction. `'minimize'` (default) means the goal is to minimize the return value, `'maximize'` means the goal is to maximize the return value.
  - `sampler` (Sampler, optional): Specify the sampling algorithm. Default is `TPESampler` (i.e., TPE algorithm).
  - `pruner` (Pruner, optional): Specify a pruner for early termination of trials without promise.
  - `study_name` (str, optional): The name of the study, useful when storing the study.
  - `storage` (str or None, optional): Database URL for storing the study.

**Available Samplers Explanation:**

- **`TPESampler`**: Default sampler, a Bayesian optimization algorithm based on the Tree-structured Parzen Estimator.
  ```python
  # Complete configuration example
  from optuna.samplers import TPESampler
  sampler = TPESampler(
    seed=42,                    # Random seed
    n_startup_trials=10,        # Number of initial random sampling trials
    multivariate=True,          # Whether to use multivariate TPE
    prior_weight=1.0,           # Weight of the prior distribution
    consider_magic_clip=True,   # Use magic clip to stabilize kernel density estimation
    consider_endpoints=True,    # Whether to consider endpoints in kernel density estimation
    n_ei_candidates=24          # Number of candidate points in EI maximization
  )
  ```

- **`RandomSampler`**: Pure random search sampler, similar to scikit-learn's RandomizedSearchCV.
  ```python
  from optuna.samplers import RandomSampler
  sampler = RandomSampler(seed=42)
  ```

- **`CmaEsSampler`**: Uses the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm, particularly suitable for optimizing continuous parameters.
  ```python
  from optuna.samplers import CmaEsSampler
  sampler = CmaEsSampler(
    seed=42,
    x0=None,           # Initial mean vector
    sigma0=0.1,        # Initial step size
    n_startup_trials=1 # Number of random trials before starting CMA-ES
  )
  ```

- **`NSGAIISampler`**: Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective optimization.
  ```python
  from optuna.samplers import NSGAIISampler
  sampler = NSGAIISampler(
    seed=42,
    population_size=50,  # Number of individuals per generation
    crossover_prob=0.9,  # Crossover probability
    mutation_prob=None   # Mutation probability
  )
  ```

- **`GridSampler`**: Traditional grid search sampler, will traverse all parameter combinations.
  ```python
  from optuna.samplers import GridSampler
  search_space = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
  }
  sampler = GridSampler(search_space)
  ```

- **`BruteForceSampler`**: Brute force sampler, used to enumerate all possible discrete parameter combinations.

- **`PartialFixedSampler`**: Sampler with partially fixed parameters, used for conditional search spaces.

- **`MOTPESampler`**: Multi-objective Tree-structured Parzen Estimator sampler, used for multi-objective optimization.

### Step 4: View Optimization Results
After optimization is complete, all results are saved in the `study` object and can be conveniently accessed through its attributes and methods.

```python
# 4. View optimization results
print("\n" + "="*50)
print("Optimization completed")
print("="*50)

# study.best_trial contains all information about the best trial
best_trial = study.best_trial
print(f"Best loss value (1 - accuracy): {best_trial.value:.4f}")
print(f"Corresponding cross-validation accuracy: {1 - best_trial.value:.4f}")

# study.best_params directly returns a dictionary of the best hyperparameters
print("Best hyperparameter combination found:")
print(study.best_params)

# Visualize optimization history
import optuna.visualization as vis
import matplotlib.pyplot as plt

# Plot optimization history
vis.plot_optimization_history(study).show()

# Plot parameter importance
vis.plot_param_importances(study).show()

# Plot correlation between parameters
vis.plot_contour(study).show()
```

**`study` Object Result Attributes Explanation:**

- **`study.best_trial`**: Returns a `FrozenTrial` object containing all information about the best trial (such as parameters, values, start/end times, etc.).
- **`study.best_value`**: Directly returns the objective function value of the best trial (in this case, the minimum `loss`).
- **`study.best_params`**: Returns a dictionary containing the hyperparameter combination of the best trial. This is one of the most commonly used results, and its values are already of the correct type (integers are integers), no manual conversion needed.
- **`study.trials`**: Returns a list containing all completed trial objects.
- **`study.trials_dataframe()`**: Converts all trial history to a Pandas DataFrame, very convenient for in-depth analysis and visualization.
- **`study.get_trials(deepcopy=True, states=None)`**: Gets trials that meet specific states.
- **`study.direction`**: Returns the optimization direction ('minimize' or 'maximize').


## Conclusion

Bayesian optimization provides a powerful and rigorous mathematical framework for solving high-cost black-box optimization problems. It constructs a surrogate model (such as GPR or TPE) to approximate the true objective function and uses an acquisition function to intelligently balance exploration and exploitation, thereby efficiently finding the optimal solution. GPR provides a solid theoretical foundation for Bayesian optimization, while TPE demonstrates excellent performance and scalability in handling complex, high-dimensional practical problems.
