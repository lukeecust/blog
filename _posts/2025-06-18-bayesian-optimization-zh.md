---
title: 贝叶斯优化-从数学原理到超参数调优
author: lukeecust
date: 2025-06-18 23:09:00 +0800
categories: [Machine Learning, Hyperparameter Optimization]
tags: [optimization, bayesian, HPO]
lang: zh
math: true
translation_id: bayesian-optimization
permalink: /zh/posts/bayesian-optimization/
render_with_liquid: false
---

在机器学习领域，模型性能在很大程度上取决于超参数（Hyperparameters）的选择。然而，寻找最优超参数组合的过程——即超参数优化（Hyperparameter Optimization, HPO）——始终是一项具有挑战性的任务。本文将聚焦当前该领域的主流方法之一：贝叶斯优化（Bayesian Optimization, BO）。我们将从其数学原理出发，深入解析贝叶斯优化的核心组成部分，并通过 `hyperopt` 库演示其在实际应用中的效果。

## 超参数优化：一个黑盒优化问题

机器学习模型的超参数需要在训练开始前设定。其与模型性能之间的关系往往是：

*   **表达式未知（Black-box）**：我们无法写出模型性能（如验证集准确率）关于超参数的显式数学函数 $f(x)$。我们只能通过一次完整的训练和评估过程，获得一个特定超参数组合 $x$ 对应的性能得分 $f(x)$，且无法获取其梯度信息。
*   **评估成本高昂（Expensive）**：每次评估 $f(x)$ 都可能需要数小时甚至数天。
*   **高维且复杂**：搜索空间维度高，且可能包含连续、离散和条件参数。

传统的网格搜索和随机搜索因未有效利用历史信息而效率低下。贝叶斯优化作为一种**序列化基于模型的优化（Sequential Model-Based Optimization, SMBO）**策略，通过构建目标函数的概率模型，智能地选择下一个评估点，从而大大提升了优化效率。

## 贝叶斯优化的数学框架

贝叶斯优化的目标是求解黑盒函数 $f(x)$ 的全局极值点（以最大化为例）：

$$
\begin{equation}
x^* = \underset{x \in \mathcal{X}}{\text{argmax}} \, f(x)  
\end{equation}
$$

其中 $\mathcal{X}$ 是超参数的搜索空间。算法的核心由两个关键组件构成：

1.  **概率代理模型（Probabilistic Surrogate Model）**：逼近真实黑盒函数 $f(x)$ 的简单模型，能预测任意点的函数值并量化其**不确定性**。
2.  **采集函数（Acquisition Function）**：基于代理模型的预测，构造一个函数来评估在每个候选点进行下一次采样的“价值”，并找到其最大值点作为下一个采样点。

算法的迭代流程如下：
1.  **初始化**：随机采样少量点，构成初始数据集 $D_0 = \{(x_1, y_1), ..., (x_n, y_n)\}$。
2.  **循环迭代** for $t=1, 2, ...$:
    a.  使用当前数据 $$D_{t-1}$$ 拟合概率代理模型，更新对 $$f(x)$$ 的后验概率分布。
    b.  优化采集函数 $$\alpha(x)$$，找到下一个评估点：$$x_t = \underset{x \in \mathcal{X}}{\text{argmax}} \, \alpha(x)$$。
    c.  评估 $$y_t = f(x_t)$$。
    d.  更新数据集 $$D_t = D_{t-1} \cup \{(x_t, y_t)\}$$。
3.  **终止**后，从所有已评估的点中返回最优者。

###  组件一：概率代理模型 - 学习目标函数

代理模型是贝叶斯优化的核心。它根据已有的观测点，在整个搜索空间上给出一个关于目标函数值的概率分布。最主流的两种方法是高斯过程回归（GPR）和树状结构Parzen估计器（TPE）。

####  高斯过程回归 (GPR)：经典理论基石

**高斯过程（Gaussian Process, GP）** 假设任意一组输入点对应的函数值的联合分布都服从一个多元高斯分布。一个 GP 由其**均值函数** $m(x)$ 和**协方差函数（核函数）** $k(x, x')$ 定义：
$$
\begin{equation}
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
\end{equation}
$$
**高斯过程回归（GPR）** 利用 GP 先验和已观测数据 $D_t$，可以推导出在任意新点 $$x_\ast$$ 处的后验分布，该分布仍为高斯分布 $$p(f(x_\ast) | D_t, x_\ast) = \mathcal{N}(\mu_t(x_\ast), \sigma_t^2(x_\ast))$$，其均值和方差有解析解：
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
其中 $\mathbf{K}$ 是核矩阵，$\mathbf{k}_\ast$ 是新点与观测点间的核向量，$\mathbf{y}$ 是观测值向量。$\mu_t(x_\ast)$ 是对 $f(x_\ast)$ 的最佳预测，而 $\sigma_t^2(x_\ast)$ 则量化了该预测的不确定性。

####  树状结构Parzen估计器 (TPE)：实用的高效替代方案

与 GPR 直接建模 $p(y|x)$ 不同，**TPE（Tree-structured Parzen Estimator）** 通过贝叶斯定理转而对 $p(x|y)$ 和 $p(y)$ 进行建模。其核心思想是：

1.  **划分数据**：根据一个分位数阈值 $y^\ast$（例如，取所有观测值的最好15%），将历史观测数据划分为“好”的一组 $\mathcal{D}_g = \{(x,y) | y < y^\ast\}$ 和“坏”的一组 $\mathcal{D}_b = \{(x,y) | y \ge y^\ast\}$。
2.  **建立密度模型**：分别为这两组数据的超参数 $x$ 建立概率密度模型。好的参数分布为 $l(x) = p(x|y<y^\ast)$，坏的参数分布为 $g(x) = p(x|y \ge y^\ast)$。这些密度函数通常使用Parzen窗（即核密度估计）来估计。
3.  **优化采集函数**：TPE的采集函数与期望改进量（EI）相关，最终目标是寻找使比值 $l(x)/g(x)$ 最大化的点 $x$。直观上，这意味着我们要寻找那些在“好”的分布中概率很高，但在“坏”的分布中概率很低的的参数点。

TPE的主要优势在于它能自然地处理复杂的、包含条件和离散变量的树状结构搜索空间，并且计算上比GPR更具扩展性。

| 特性 | 高斯过程回归 (GPR) | 树状结构Parzen估计器 (TPE) |
| :--- | :--- | :--- |
| **核心思想** | 建模 $$p(y|x)$$，直接估计函数值的分布 | 建模 $$p(x|y)$$，估计参数在好/坏情况下的分布 |
| **数学基础** | 高斯过程、贝叶斯线性回归 | 贝叶斯定理、核密度估计（Parzen窗） |
| **参数空间** | 最适合**连续**和**低维**空间 | 极佳地处理**离散、条件参数**和**高维**空间 |
| **计算复杂度** | $O(n^3)$，受限于核矩阵求逆，扩展性较差 | $O(n \log n)$，扩展性更好 |
| **并行性** | 难以并行，本质上是序列化的 | 更易于并行化，可通过从 $l(x)$ 采样生成多个候选点 |
| **常见工具** | `scikit-optimize`, `GPyOpt` | `hyperopt`, `Optuna` |

### 组件二：采集函数 - 平衡探索与利用

采集函数 $\alpha(x)$ 利用代理模型的预测 $(\mu(x), \sigma(x))$ 来决定下一个采样点，它必须巧妙地平衡**利用（Exploitation）**和**探索（Exploration）**。令 $f^+ = \max_{i} f(x_i)$ 为当前观测到的最优值。

1.  **Probability of Improvement (PI)**：
    $$
    \begin{equation}
    \alpha_{PI}(x) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)
    \end{equation}
    $$
    其中 $\Phi(\cdot)$ 是标准正态CDF，$\xi$ 是调节因子。

2.  **Expected Improvement (EI)**：最常用的采集函数，计算期望的改进量。
    $$\begin{equation}
    \alpha_{EI}(x) = (\mu(x) - f^+) \Phi(Z) + \sigma(x)\phi(Z) \quad \text{where} \quad Z = \frac{\mu(x) - f^+}{\sigma(x)}
    \end{equation}$$
    其中 $\phi(\cdot)$ 是标准正态PDF。

3.  **Upper Confidence Bound (UCB)**：
    $$\begin{equation}
    \alpha_{UCB}(x) = \mu(x) + \kappa \sigma(x)
    \end{equation}$$
    其中 $\kappa \ge 0$ 控制探索的权重。

## AutoML, HPO, NAS

在讨论贝叶斯优化时，有必要厘清其在自动化机器学习（AutoML）生态中的位置：

*   **AutoML**：一个宏大的目标，旨在将机器学习的全流程自动化，包括数据预处理、特征工程、模型选择、超参数优化（HPO）和神经架构搜索（NAS）。
*   **HPO vs. NAS**：两者都属于AutoML的范畴，但优化的对象不同。

| 项目 | HPO (Hyperparameter Optimization) | NAS (Neural Architecture Search) |
| :--- | :--- | :--- |
| **优化对象** | 模型的超参数（如学习率、batch size、正则化系数） | 模型的网络结构（如层数、卷积核大小、连接方式） |
| **搜索空间** | 通常是标量/连续/离散值的组合 | 通常是图结构、模块组合等离散且复杂的空间 |
| **优化方法** | 网格/随机搜索, **贝叶斯优化**, 进化算法等 | 强化学习, 进化算法, **贝叶斯优化**, 梯度方法等 |

贝叶斯优化是实现 HPO 的一种先进且高效的方法，有时也被应用于 NAS 任务中。

## 使用 `hyperopt` (TPE) 进行 XGBoost 调优

`hyperopt` 是一个实现了SMBO框架的流行Python库，其核心算法是TPE。下面，我们用它来为一个 `XGBoost` 分类器寻找最优超参数。

### 第1步：定义目标函数 $f(x)$
```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2. 定义目标函数
def objective(params):
    # hyperopt 会传递浮点数，某些参数需要转为整数
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
    clf = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # 使用交叉验证评估模型，返回负的准确率作为损失
    accuracy = cross_val_score(clf, X, y, cv=5).mean()
    loss = 1 - accuracy
    
    # hyperopt 需要一个字典返回，必须包含 'loss' 和 'status'
    return {'loss': loss, 'status': STATUS_OK, 'accuracy': accuracy}
```

### 第2步：定义搜索空间 $\mathcal{X}$
```python
# 3. 定义搜索空间
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    'subsample': hp.uniform('subsample', 0.5, 1.0)
}
```
**参数说明：**

- **`hp.quniform(label, low, high, q)`**: 离散均匀分布
  - `label`: 参数标签名称
  - `low`: 最小值
  - `high`: 最大值
  - `q`: 离散化步长

- **`hp.uniform(label, low, high)`**: 连续均匀分布
  - `label`: 参数标签名称
  - `low`: 最小值
  - `high`: 最大值

- **`hp.loguniform(label, low, high)`**: 对数均匀分布
  - `label`: 参数标签名称
  - `low`: 最小值的对数
  - `high`: 最大值的对数

- **其他可用分布函数：**
  - `hp.choice(label, options)`: 从选项中选择
  - `hp.randint(label, upper)`: 随机整数 [0, upper)
  - `hp.normal(label, mu, sigma)`: 正态分布
  - `hp.lognormal(label, mu, sigma)`: 对数正态分布
  
### 第3步：运行贝叶斯优化
```python
# 4. 运行优化
trials = Trials()

best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # 明确指定使用TPE算法
    max_evals=100,
    trials=trials
)

print("\n" + "="*50)
print("优化完成")
print("="*50)
# fmin返回的是使loss最小的参数值，但某些值可能是浮点数，需要整理
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
print("找到的最佳超参数组合:")
print(best_params)

# 从trials对象中获取最佳的试验结果
best_trial = trials.best_trial
print(f"\n最佳损失值 (1 - accuracy): {best_trial['result']['loss']:.4f}")
print(f"对应的交叉验证准确率: {best_trial['result']['accuracy']:.4f}")
```


**参数说明：**

- **`fmin()`** 参数：
  - `fn`: 目标函数 (必需)
  - `space`: 搜索空间 (必需)
  - `algo`: 优化算法 (必需，如 `tpe.suggest`)
  - `max_evals`: 最大评估次数 (必需)
  - `trials`: 试验对象 (可选，默认为None)
  - `rstate`: 随机状态 (可选，默认为None)
  - `verbose`: 详细输出 (可选，默认为0)
  - `return_argmin`: 是否返回最小值参数 (可选，默认为True)
  - `show_progressbar`: 是否显示进度条 (可选，默认为True)

- **`Trials()`** 参数：
  - `exp_key`: 实验关键字 (可选，默认为None)
  - `refresh`: 是否刷新 (可选，默认为True)

- **其他可用算法：**
  - `tpe.suggest`: TPE算法 (Tree-structured Parzen Estimator)
  - `rand.suggest`: 随机搜索
  - `anneal.suggest`: 模拟退火

## 结论

贝叶斯优化为解决高成本的黑盒优化问题提供了一个强大而严谨的数学框架。它通过构建代理模型（如GPR或TPE）来逼近真实目标函数，并利用采集函数智能地平衡探索与利用，从而高效地找到最优解。GPR为贝叶斯优化提供了坚实的理论基础，而TPE则在处理复杂、高维的实际问题中展现出卓越的性能和扩展性。







