---
title: Huber Loss Function (Smooth L1 Loss)
author: lukeecust
date: 2025-06-21 02:09:00 +0800
categories: [Deep Learning, LossFunction]
tags: [Regression, HuberLoss, SmoothL1]
lang: en
math: true
translation_id: huber-loss
permalink: /posts/huber-loss/
render_with_liquid: false
---


In machine learning regression tasks, we're always searching for the perfect loss function to guide model learning. The two most common choices are Mean Squared Error (MSE, L2 Loss) and Mean Absolute Error (MAE, L1 Loss). But they're like two sides of a coin, each with its own strengths and weaknesses:

*   **MSE** squares the errors, making it smooth near the optimum with stable convergence. But this same property is also its weakness—when encountering outliers, a huge error gets squared, producing an extremely large loss value that can "hijack" the entire model, making it deviate from the trend of normal data.
*   **MAE** treats all errors equally, only taking absolute values. This makes it naturally insensitive to outliers, very "robust." But its derivative is discontinuous at zero, which creates problems for gradient descent—near the optimum, the gradient fluctuates between positive and negative values, potentially causing the optimization process to "swing" back and forth, making it difficult to converge precisely to the minimum.

Is there a way to combine the strengths of both approaches, being both robust against outliers and smooth near the optimum? This is where today's protagonist comes in—the **Huber Loss Function**, also commonly known as **Smooth L1 Loss**.

## **Huber Loss Function Definition**

Simply put, the Huber loss function is a "two-faced" function:

> When the error is small, it behaves like MSE; when the error is large, it switches to MAE mode.

This boundary between "large" and "small" is determined by a hyperparameter $\delta$ that we set ourselves.

Assuming the predicted value is $\hat{y}$, the true value is $y$, then the error is $e = y - \hat{y}$. The Huber loss $L_{\delta}(e)$ is calculated as follows:

$$\begin{equation}
L_\delta(e)= \begin{cases}\frac{1}{2} e^2, & \text{ if } |e| \le \delta \\ \delta\left(|e|-\frac{1}{2} \delta\right), & \text{ if } |e| > \delta\end{cases}
\end{equation}$$

1.  **When the absolute value of the error $\lvert e \rvert$ is less than or equal to $\delta$**: The loss function is $\frac{1}{2}e^2$. This is the form of MSE. In this interval, we consider the error to be "normal," and using quadratic penalties allows the model to make finer adjustments as it approaches the optimal solution.
2.  **When the absolute value of the error $\lvert e \rvert$ is greater than $\delta$**: The loss function becomes $\delta(\lvert e \rvert - \frac{1}{2}\delta)$. This is a linear function, growing similarly to MAE. This means that when an error is large enough to be considered an "outlier," we only give it a linear penalty, avoiding its disproportionately huge impact on the total loss.

Most cleverly, this function is not only continuous at the boundary point $e = \pm\delta$, but its derivative is also continuous. This solves the problem of MAE not being differentiable at zero, ensuring that gradient-based optimization algorithms can run smoothly.

## **Hyperparameter $\delta$: The "Arbitrator" of Outliers**

The choice of $\delta$ is crucial, as it directly defines how the model views errors: which ones are "noise" that needs to be fitted precisely, and which ones are "outliers" that should be tolerated.

*   **When $\delta \to 0$**, any tiny error is considered a "large error," and the behavior of Huber loss approaches that of **MAE**.
*   **When $\delta \to \infty$**, almost all errors are considered "small errors," and the behavior of Huber loss approaches that of **MSE**.

In practice, the optimal value of $\delta$ is usually determined through cross-validation. However, Huber himself, in his original paper, gave a classic recommended value of $\delta = 1.345\sigma$ (where $\sigma$ is the standard deviation). For data with a standard normal distribution, setting $\delta=1.345$ allows the model to maintain robustness against outliers while still achieving 95% of the statistical efficiency that MSE would have in ideal (outlier-free) conditions. This is a very useful empirical starting point.

## **Why Do We Need Huber Loss?**

1.  **Combines the advantages of MSE and MAE**:
    *   **Robust against outliers**: Inherits the characteristic of MAE, where the loss grows linearly when the error exceeds $\delta$, effectively suppressing the excessive influence of outliers on model training.
    *   **Stable convergence near the optimal solution**: Inherits the characteristic of MSE, where the loss is a smooth quadratic function when the error is less than $\delta$. The gradient decreases as the error decreases, helping the model converge more precisely to the minimum value, avoiding the gradient oscillation problem of MAE near the optimal solution.

2.  **Theoretical Background: Dealing with Heavy-tailed Distributions**
    Why are we so concerned about outliers? In statistics, many real-world data distributions are not ideal Gaussian distributions. They might be **heavy-tailed distributions**, meaning data points have a higher probability of appearing far from the mean. These "distant" data points are what we commonly refer to as "distributional outliers." For such data, using MSE, which is sensitive to outliers, would cause severe model distortion, and Huber loss is precisely the robust estimation method designed for this situation.

## **Advantages and Disadvantages of Huber Loss**

**Advantages:**
*   **Enhanced robustness against outliers**, solving the problem of MSE being sensitive to outliers.
*   **Resolves the issue of MAE's optimization instability near the optimal solution**, making the training process smoother.
*   **Convergence speed is typically faster than MAE**, because it utilizes the quadratic descent characteristic of MSE when the error is small.

**Disadvantages:**
*   Introduces **an additional hyperparameter $\delta$**, requiring us to adjust it through methods such as cross-validation, which increases the workload of parameter tuning.

## **Code Implementation of Huber Loss**

### NumPy Implementation

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss
    y_true: True values
    y_pred: Predicted values
    delta: Threshold for switching between MSE and MAE behavior
    """
    diff = np.abs(y_true - y_pred)
    
    # Use np.where for conditional judgment and calculation
    loss = np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    
    return np.mean(loss)

# Example
y_true = np.array([1.0, 2.0, 3.0, 10.0]) # The last one is an outlier
y_pred = np.array([1.1, 2.2, 2.9, 5.0])

loss_val = huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {loss_val}")
```
### Scikit-learn Implementation

Scikit-learn provides a regression model based on Huber loss called `HuberRegressor`, which is specifically designed for datasets containing outliers:

```python
from sklearn.linear_model import HuberRegressor
import numpy as np

# Prepare data
X = np.array([[1], [2], [3], [4]])
y = np.array([1, 2, 3, 10])  # The last one is an outlier

# Create and train HuberRegressor model
huber = HuberRegressor(epsilon=1.35)  # epsilon is equivalent to the delta parameter
huber.fit(X, y)

# Prediction
y_pred = huber.predict(X)
print("Predictions:", y_pred)
print("Model coefficients:", huber.coef_)
print("Model intercept:", huber.intercept_)
```

Important parameters of `HuberRegressor`:
- `epsilon`: Similar to the δ parameter we discussed, determines the switching point between MSE and MAE behavior
- `alpha`: Regularization parameter, controls the strength of L2 regularization
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance

###  PyTorch Implementation
```python
import torch
import torch.nn as nn

# Simulate true values and predicted values
y_true = torch.tensor([1.5, 2.0, 3.0])
y_pred = torch.tensor([1.0, 2.5, 2.5])

# Define Huber loss
loss_fn = nn.HuberLoss(reduction='mean')

loss = loss_fn(y_pred, y_true)
print(f"Huber Loss: {loss.item()}")
```
Parameter explanation: `reduction='mean'`
* `'none'`: Returns the loss for each sample without aggregation;
* `'mean'`: Returns the average of all losses (default);
* `'sum'`: Returns the sum of all losses.


## **Conclusion**

The Huber loss function is a veritable "Swiss Army knife" in the machine learning toolbox. Through a clever piecewise design, it achieves a perfect balance between the smooth optimization of MSE and the robustness of MAE. Although it introduces an additional hyperparameter $\delta$, when dealing with real-world data full of noise and outliers, this small cost of parameter tuning often leads to a qualitative leap in model performance.