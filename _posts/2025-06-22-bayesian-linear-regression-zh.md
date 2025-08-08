---
title: 贝叶斯线性回归
author: lukeecust
date: 2025-06-22 15:09:00 +0800
categories: [Machine Learning, Model, Linear Regression]
tags: [LR, bayesian]
lang: zh
math: true
translation_id: bayesian-linear-regression
permalink: /zh/posts/bayesian-linear-regression/
render_with_liquid: false
---

当我们提到“线性回归”时，脑海中浮现的通常是**最小二乘法 (OLS)**——那条穿过数据点的“最佳拟合”直线。OLS 简单直观，但它给出的答案是唯一的、确定的。它会告诉你：“最佳的斜率是 2.5”，却无法回答：“这个斜率有多大可能是 2.4 或 2.6？我们对这个估计的信心有多大？”

这就是**贝叶斯线性回归 (Bayesian Linear Regression)** 的用武之地。它将我们从寻找单一“最佳”值的世界，带入一个充满可能性的概率世界。它不仅给出预测，更重要的是，它量化了**不确定性**。


## 核心思想：从点估计到概率分布

贝叶斯方法的核心转变在于它如何看待模型参数（例如，权重 $\mathbf{w}$）。

*   **频率派（如 OLS）**：认为参数 $\mathbf{w}$ 是一个未知但**固定**的常量。我们的目标是找到它的最佳点估计。
*   **贝叶斯派**：认为参数 $\mathbf{w}$ 本身就是一个**随机变量**，它遵循一个概率分布。

因此，我们的目标不再是找到单一的 $\mathbf{w}$，而是根据观测到的数据，去推断 $\mathbf{w}$ 的**后验概率分布 (Posterior Distribution)**。整个过程遵循贝叶斯定理：

$$\begin{equation}
    p(\text{参数} \vert \text{数据}) = \frac{p(\text{数据} \vert \text{参数}) \times p(\text{参数})}{p(\text{数据})}
\end{equation}$$
转换成我们线性回归的术语：

$$\begin{equation}
p(\mathbf{w} \vert \mathcal{D}) \propto p(\mathcal{D} \vert \mathbf{w}) \times p(\mathbf{w})
\end{equation}$$

这里：
*   $p(\mathbf{w} \vert \mathcal{D})$ 是**后验概率 (Posterior)**：在看到数据 $\mathcal{D}$ 后，我们对参数 $\mathbf{w}$ 的信念。这是我们最终想要求解的。
*   $p(\mathcal{D} \vert \mathbf{w})$ 是**似然 (Likelihood)**：假设模型为 $y \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}, \sigma^2)$，似然描述了在给定参数 $\mathbf{w}$ 的情况下，当前数据出现的可能性。
*   $p(\mathbf{w})$ 是**先验概率 (Prior)**：在看到任何数据之前，我们对参数 $\mathbf{w}$ 的初始信念。这是贝叶斯方法的一大优势，它允许我们**融入领域知识**。

那么，这个后验分布具体是如何计算出来的呢？让我们进入数学的世界。

## 数学原理：求解后验分布

为了求解后验，我们需要先明确定义似然和先验的形式。

### 似然函数 $p(\mathcal{D} \vert \mathbf{w})$

我们假设模型输出 $y$ 与真实值 $\mathbf{w}^T \mathbf{x}$ 之间存在高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。为了数学上的方便，我们通常使用**精度 (precision)** $\beta = 1/\sigma^2$ 来表示。因此，对于单个数据点，其概率为：
$$\begin{equation}
p(y_i \vert \mathbf{x_i}, \mathbf{w}, \beta) = \mathcal{N}(y_i \vert \mathbf{w}^T \mathbf{x_i}, \beta^{-1})
\end{equation}$$

对于整个独立同分布的数据集 $\mathcal{D} = \{(\mathbf{X}, \mathbf{y})\}$，似然函数是所有数据点概率的乘积：
$$\begin{equation}
p(\mathbf{y} \vert \mathbf{X}, \mathbf{w}, \beta) = \prod_{i=1}^{N} \mathcal{N}(y_i \vert \mathbf{w}^T \mathbf{x_i}, \beta^{-1})
\end{equation}$$

### 先验分布 $p(\mathbf{w})$

为了让计算变得可行，我们为先验选择一个**共轭先验 (Conjugate Prior)**。当先验和似然是共轭的时候，它们的乘积（即后验）将和先验具有相同的函数形式。对于高斯似然，其共轭先验也是高斯分布。我们假设 $\mathbf{w}$ 服从一个均值为 $\mathbf{m}_0$，协方差为 $\mathbf{S}_0$ 的多维高斯分布：
$$\begin{equation}
p(\mathbf{w}) = \mathcal{N}(\mathbf{w} \vert \mathbf{m}_0, \mathbf{S}_0)
\end{equation}$$

通常，我们会选择一个简单的零均值先验，$\mathbf{m}_0 = \mathbf{0}$，协方差矩阵为 $\mathbf{S}_0 = \alpha^{-1} \mathbf{I}$，其中 $\alpha$ 是一个超参数，代表我们对 $\mathbf{w}$ 权重大小的先验信念精度。

###  推导后验分布 $p(\mathbf{w} \vert \mathcal{D})$

现在，我们将似然和先验相乘。为了简化，我们处理它们的对数形式，并忽略与 $\mathbf{w}$ 无关的常数项：
$$\begin{equation}
\begin{aligned}
\ln p(\mathbf{w} \vert \mathcal{D}) &\propto \ln p(\mathbf{y} \vert \mathbf{X}, \mathbf{w}, \beta) + \ln p(\mathbf{w}) \\
&= \ln \left[ \exp(-\frac{\beta}{2} (\mathbf{y} - \mathbf{Xw})^T (\mathbf{y} - \mathbf{Xw})) \right] + \ln \left[ \exp(-\frac{1}{2} (\mathbf{w} - \mathbf{m}_0)^T \mathbf{S}_0^{-1} (\mathbf{w} - \mathbf{m}_0)) \right] \\
&= -\frac{\beta}{2} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{X}^T\mathbf{Xw}) - \frac{1}{2} (\mathbf{w}^T\mathbf{S}_0^{-1}\mathbf{w} - 2\mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{w} + \mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{m}_0) + \text{const}
\end{aligned}
\end{equation}$$

我们的目标是把这个式子整理成一个关于 $\mathbf{w}$ 的标准高斯分布的对数形式：$-\frac{1}{2}(\mathbf{w}-\mathbf{m}_N)^T \mathbf{S}_N^{-1} (\mathbf{w}-\mathbf{m}_N)$。我们只关注包含 $\mathbf{w}$ 的项：
*   **$\mathbf{w}$ 的二次项**：$-\frac{1}{2}(\beta \mathbf{w}^T\mathbf{X}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{S}_0^{-1}\mathbf{w}) = -\frac{1}{2}\mathbf{w}^T(\beta \mathbf{X}^T\mathbf{X} + \mathbf{S}_0^{-1})\mathbf{w}$
*   **$\mathbf{w}$ 的一次项**：$\beta \mathbf{y}^T\mathbf{Xw} + \mathbf{m}_0^T\mathbf{S}_0^{-1}\mathbf{w} = (\beta \mathbf{y}^T\mathbf{X} + \mathbf{m}_0^T\mathbf{S}_0^{-1})\mathbf{w}$

通过对比标准高斯对数形式展开后的二次项 $-\frac{1}{2}\mathbf{w}^T\mathbf{S}_N^{-1}\mathbf{w}$ 和一次项 $\mathbf{m}_N^T\mathbf{S}_N^{-1}\mathbf{w}$，我们可以得到：
$$\begin{equation}
\mathbf{S}_N^{-1} = \mathbf{S}_0^{-1} + \beta \mathbf{X}^T \mathbf{X}
\end{equation}$$
$$\begin{equation}
\mathbf{m}_N^T\mathbf{S}_N^{-1} = \mathbf{y}^T(\beta\mathbf{X}^T) + \mathbf{m}_0^T\mathbf{S}_0^{-1}
\end{equation}$$

从第一个式子，我们得到后验协方差 $\mathbf{S}_N$：
$$\begin{equation}
\mathbf{S}_N = (\mathbf{S}_0^{-1} + \beta \mathbf{X}^T \mathbf{X})^{-1}
\end{equation}$$

将 $\mathbf{S}_N^{-1}$ 代入第二个式子并求解 $\mathbf{m}_N$，我们得到后验均值 $\mathbf{m}_N$：
$$\begin{equation}
\mathbf{m}_N = \mathbf{S}_N (\mathbf{S}_0^{-1} \mathbf{m}_0 + \beta \mathbf{X}^T \mathbf{y})
\end{equation}$$


通过数学推导，得到了一个全新的高斯分布 $\mathcal{N}(\mathbf{w} \vert \mathbf{m}_N, \mathbf{S}_N)$。这就是我们根据数据更新后的、对参数 $\mathbf{w}$ 的最终信念！

###  预测新数据点

当我们有一个新的数据点 $$\mathbf{x}_{\ast}$$ 时，预测值 $$y_{\ast}$$ 的分布可以通过对所有可能的 $$\mathbf{w}$$ 进行积分得到：
$$\begin{equation}
p(y_{\ast} \vert \mathbf{x}_{\ast}, \mathcal{D}) = \int p(y_{\ast} \vert \mathbf{x}_{\ast}, \mathbf{w}) p(\mathbf{w} \vert \mathcal{D}) d\mathbf{w}
\end{equation}$$

这个积分的结果也是一个高斯分布，其均值为 $$\mathbf{m}_N^T \mathbf{x}_{\ast}$$，方差为：
$$\begin{equation}
\sigma_{\text{pred}}^2 = \underbrace{\frac{1}{\beta}}_{\text{数据固有噪声}} + \underbrace{\mathbf{x}_{\ast}^T \mathbf{S}_N \mathbf{x}_{\ast}}_{\text{模型参数不确定性}}
\end{equation}$$


这个预测方差完美地诠释了贝叶斯方法的精髓：**总不确定性 = 数据本身的随机性 + 我们对模型认知的不确定性。**

## 两种实现路径：解析解 vs. 采样

上述数学推导为我们提供了一个**解析解**。只要我们选择共轭的先验和似然，就能直接计算出后验分布。
1.  **解析/近似方法 (Analytical/Approximate)**：
    这种方法高效、快速。Scikit-learn 的 `BayesianRidge` 就是基于这种思想，通过优化来找到后验分布的近似解。

2.  **马尔可夫链蒙特卡洛 (MCMC) 采样**：
    但如果我们想用更复杂的先验（比如一个双峰分布），或者模型的似然不是高斯分布呢？解析解就不再存在了。这时，我们就需要 MCMC 这样的采样技术，从难以直接计算的后验分布中抽取成千上万个样本来近似它。`PyMC` 就是为此而生的强大工具。

现在，让我们看看这两种路径在代码中是如何实现的。

## 代码实现

### Scikit-learn 的 `BayesianRidge` 

`BayesianRidge` 应用了我们上面推导的原理，并使用**经验贝叶斯**方法自动估计超参数 $\alpha$ 和 $\beta$。它非常适合作为标准线性回归的直接替代品。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 1. 生成模拟数据
def generate_data(n_samples=30, noise_std=0.5):
    np.random.seed(42)
    X = np.linspace(-5, 5, n_samples)
    # 真实参数
    true_w = 0.5
    true_b = -1
    y = true_w * X + true_b + np.random.normal(0, noise_std, size=n_samples)
    return X.reshape(-1, 1), y

# 生成数据
X_train, y_train = generate_data()

# 2. 创建并训练模型
# BayesianRidge 会自动估计 alpha (权重精度) 和 lambda (噪声精度)
br = BayesianRidge(compute_score=True)
br.fit(X_train, y_train)

print(f"Scikit-learn BayesianRidge:")
print(f"Estimated weights (w): {br.coef_[0]:.4f}")
print(f"Estimated intercept (b): {br.intercept_:.4f}")
print(f"Estimated alpha (precision of weights): {br.alpha_:.4f}")
print(f"Estimated lambda (precision of noise): {br.lambda_:.4f}")

# 3. 创建测试点并预测
X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
y_mean_sk, y_std_sk = br.predict(X_test, return_std=True)

# 4. 可视化结果
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
**结果分析**：`BayesianRidge` 快速给出了一个带有不确定性区间的预测。这个不确定性区间（紫色阴影）的宽度，其背后的数学基础正是我们刚刚推导的预测方差公式。

**主要参数说明：**
- `n_iter`：最大迭代次数，默认为 300
- `tol`：收敛判定阈值，默认为 1e-3
- `alpha_1`、`alpha_2`：$\alpha$（权重精度）的伽马先验参数，默认分别为 1e-6 和 1e-6
- `lambda_1`、`lambda_2`：$\beta$（噪声精度）的伽马先验参数，默认分别为 1e-6 和 1e-6
- `compute_score`：是否计算每次迭代的对数边际似然，默认为 False
- `fit_intercept`：是否计算截距，默认为 True
- `normalize`：已弃用，请使用 `StandardScaler`
- `copy_X`：是否复制 X，默认为 True

**预测方法参数：**
- `return_std`：如果为 True，则返回预测的标准差，默认为 False
- `return_cov`：如果为 True，则返回预测的协方差，默认为 False

### PyMC

`PyMC` 则完全不同。我们不关心是否存在解析解，而是直接向计算机**描述我们的概率模型**，然后让 MCMC 采样器去探索后验分布。

**主要函数和分布说明：**

```python
import pymc as pm
import arviz as az

# 使用与上面相同的数据
# X_train, y_train

# 1. 定义 PyMC 模型
with pm.Model() as bayesian_linear_model:
    # 定义先验分布
    # 对截距 b 使用弱信息正态先验
    b = pm.Normal('intercept', mu=0, sigma=10)
    # 对斜率 w 使用弱信息正态先验
    w = pm.Normal('slope', mu=0, sigma=10)
    # 对数据噪声的标准差 sigma 使用弱信息半正态先验 (必须为正)
    sigma = pm.HalfNormal('sigma', sigma=5)

    # 定义线性模型 (似然的均值)
    mu = w * X_train.ravel() + b

    # 定义似然分布
    # y_obs 是我们观察到的数据
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)

    # 2. 运行 MCMC 采样器
    # 采样 2000 次，预热/调整 1000 次
    trace = pm.sample(2000, tune=1000, cores=1)
    
# 3. 分析和可视化结果
print("\nPyMC Model Summary:")
az.summary(trace, var_names=['intercept', 'slope', 'sigma'])

# 4. 生成后验预测
with bayesian_linear_model:
    # 在测试点上生成后验预测样本
    post_pred = pm.sample_posterior_predictive(trace, var_names=['y_obs'], samples=1000)

# 提取预测结果
# PyMC 4.0+ 使用 .stack(sample=("chain", "draw"))
y_pred_samples = post_pred.posterior_predictive['y_obs'].stack(sample=("chain", "draw")).values.T
# 替换为 X_test
mu_test = trace.posterior['slope'].values * X_test.ravel() + trace.posterior['intercept'].values
y_mean_pm = mu_test.mean(axis=1)
# 使用 ArviZ 计算 HDI (高密度区间), 这是比标准差更稳健的贝叶斯不确定性度量
hdi_data = az.hdi(mu_test.T, hdi_prob=0.94) # 94% HDI ~ 2 std

# 5. 可视化
plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, label="Training Data", color="blue", alpha=0.7, zorder=2)
plt.plot(X_test, y_mean_pm, label="PyMC Mean Prediction", color="green", zorder=3)
plt.fill_between(X_test.ravel(), hdi_data[:,0], hdi_data[:,1],
                 color="green", alpha=0.2, label="Uncertainty (94% HDI)", zorder=1)

# 绘制几条从后验中抽样的线
for i in np.random.randint(0, mu_test.shape[1], 10):
    plt.plot(X_test, mu_test[:, i], color='gray', alpha=0.3, lw=1)


plt.title("Full Bayesian Regression (PyMC)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
**结果分析**：`PyMC` 的结果更具"贝叶斯风味"。我们得到的不是一个单一的后验分布参数，而是成千上万个从后验分布中抽取的样本（`trace` 对象）。这些样本构成了对整个后验分布的经验近似。

**PyMC 模型构建函数：**
- `pm.Model()`：创建一个贝叶斯模型容器
- `pm.Normal(name, mu, sigma)`：创建一个正态分布变量，`mu`为均值，`sigma`为标准差
- `pm.HalfNormal(name, sigma)`：创建一个半正态分布变量，用于非负参数
- `pm.sample(draws, tune, cores)`：
  - `draws`：样本数量
  - `tune`：预热迭代次数，用于调整采样器
  - `cores`：并行计算的核心数

**ArviZ 结果分析函数：**
- `az.summary(trace, var_names)`：返回后验分布的统计摘要
- `az.hdi(data, hdi_prob)`：计算高密度区间，`hdi_prob`指定概率质量（如0.94表示94%）

### `BayesianRidge` vs. `PyMC`

| 特性 | `sklearn.BayesianRidge` | `PyMC` |
| :--- | :--- | :--- |
| **数学基础** | **依赖解析解**（或其近似） | **不依赖解析解**，使用 MCMC 采样 |
| **方法论** | 经验贝叶斯 (Empirical Bayes) | 完全贝叶斯 (Fully Bayesian) |
| **灵活性** | 低：模型结构固定（高斯先验和似然） | **极高**：可自定义任意先验、似然和复杂模型结构 |
| **速度** | **非常快** | 较慢，取决于模型复杂度和数据量 |
| **易用性** | **非常简单**，API 与其他 sklearn 模型一致 | 学习曲线较陡，需要理解贝叶斯建模思想 |
| **输出信息** | 参数的后验均值和协方差矩阵 | **所有参数的完整后验分布样本** |
| **适用场景** | 快速为标准线性问题添加不确定性估计 | 复杂/非标准模型、深入研究参数不确定性、分层模型等 |

**如何选择？**

*   **选择 `BayesianRidge`**：当你的问题符合线性回归的基本假设（高斯噪声），并且你满足于由共轭先验带来的正则化效果时。它是一个高效、务实的选择，其背后是我们刚刚推导出的优美数学。
*   **选择 `PyMC`**：当你面对一个无法用简单数学公式求解的复杂模型时，或者你想对模型的每一个环节（先验、似然）都有完全的控制权时。它解放了我们，让我们不必局限于有解析解的模型，能够探索更广阔的贝叶斯建模世界。

## 总结

贝叶斯线性回归让我们从寻找单一的“最佳答案”转向了拥抱和量化不确定性。理解其背后的数学原理，我们能更深刻地认识到，`BayesianRidge` 是这一理论的精巧应用，而 `PyMC` 则是在理论无法直接求解时，为我们提供的一条强大而通用的路径。







