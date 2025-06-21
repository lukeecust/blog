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
3. 
    a.  使用当前数据 $$D_{t-1}$$ 拟合概率代理模型，更新对 $$f(x)$$ 的后验概率分布。

    b.  优化采集函数 $$\alpha(x)$$，找到下一个评估点：$$x_t = \underset{x \in \mathcal{X}}{\text{argmax}} \, \alpha(x)$$。

    c.  评估 $$y_t = f(x_t)$$。

    d.  更新数据集 $$D_t = D_{t-1} \cup \{(x_t, y_t)\}$$。
4.  **终止**后，从所有已评估的点中返回最优者。

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
其中 $$\mathbf{K}$$ 是核矩阵，$$\mathbf{k}_\ast$$ 是新点与观测点间的核向量，$$\mathbf{y}$$ 是观测值向量。$$\mu_t(x_\ast)$$ 是对 $$f(x_\ast)$$ 的最佳预测，而 $$\sigma_t^2(x_\ast)$$ 则量化了该预测的不确定性。

####  树状结构Parzen估计器 (TPE)：实用的高效替代方案

与 GPR 直接建模 $$p(y\vert x)$$ 不同，**TPE（Tree-structured Parzen Estimator）** 通过贝叶斯定理转而对 $$p(x\vert y)$$ 和 $$p(y)$$ 进行建模。其核心思想是：

1.  **划分数据**：根据一个分位数阈值 $$y^\ast$$（例如，取所有观测值的最好15%），将历史观测数据划分为"好"的一组 $$\mathcal{D}_g = \{(x,y) \vert y < y^\ast\}$$ 和"坏"的一组 $$\mathcal{D}_b = \{(x,y) \vert y \ge y^\ast\}$$。
2.  **建立密度模型**：分别为这两组数据的超参数 $$x$$ 建立概率密度模型。好的参数分布为 $$l(x) = p(x\vert y<y^\ast)$$，坏的参数分布为 $$g(x) = p(x\vert y \ge y^\ast)$$。这些密度函数通常使用Parzen窗（即核密度估计）来估计。
3.  **优化采集函数**：TPE的采集函数与期望改进量（EI）相关，最终目标是寻找使比值 $$l(x)/g(x)$$ 最大化的点 $$x$$。直观上，这意味着我们要寻找那些在"好"的分布中概率很高，但在"坏"的分布中概率很低的的参数点。

TPE的主要优势在于它能自然地处理复杂的、包含条件和离散变量的树状结构搜索空间，并且计算上比GPR更具扩展性。

| 特性 | 高斯过程回归 (GPR) | 树状结构Parzen估计器 (TPE) |
| :--- | :--- | :--- |
| **核心思想** | 建模 $p(y\vert x)$，直接估计函数值的分布 | 建模 $p(x\vert y)$，估计参数在好/坏情况下的分布 |
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


## 使用 `optuna` (TPE) 进行 XGBoost 调优

`optuna` 是一个现代化的自动超参数优化框架，它默认使用 TPE 算法，并提供了非常直观和灵活的 "Define-by-Run" API。下面，我们用它来为一个 `XGBoost` 分类器寻找最优超参数。

### 第1步：加载数据
首先需要准备好用于模型训练和评估的数据。

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import optuna

# 1. 加载数据
iris = load_iris()
X = iris.data
y = iris.target
```

### 第2步：定义目标函数 $f(x)$
这是 `optuna` 与 `hyperopt` 核心区别所在。在 `optuna` 中，搜索空间是在目标函数内部通过 `trial` 对象动态定义的。目标函数接收一个 `trial` 对象作为参数，并返回一个需要被优化的数值（如损失或准确率）。

```python
# 2. 定义目标函数
def objective(trial):
  # 在函数内部，通过 trial 对象建议(suggest)超参数的值
  # 这就是 "Define-by-Run" API
  params = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=25),
    'max_depth': trial.suggest_int('max_depth', 3, 15),
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
    'gamma': trial.suggest_float('gamma', 0, 0.5),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
    
    # 固定参数可以直接写入
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
  }
  
  clf = xgb.XGBClassifier(**params)
  
  # 使用交叉验证评估模型，返回负的准确率作为损失
  accuracy = cross_val_score(clf, X, y, cv=5).mean()
  
  # Optuna 会根据 study 的优化方向来处理这个返回值
  # 我们希望最小化损失，所以返回 1 - accuracy
  loss = 1 - accuracy
  return loss
```

**`trial` 对象方法说明：**
`trial` 对象是在目标函数内部用于定义搜索空间的核心工具。

- **`trial.suggest_int(name, low, high, step=1, log=False)`**: 建议一个整数。
  - `name` (str): 参数的名称，在一次 `study` 中必须是唯一的。
  - `low` (int): 搜索范围的下界（包含）。
  - `high` (int): 搜索范围的上界（包含）。
  - `step` (int): 建议值的步长。例如 `step=25` 会从 `[50, 75, 100, ...]` 中取值。
  - `log` (bool): 若为 `True`，则在对数尺度上采样，适用于跨数量级的整数。

- **`trial.suggest_float(name, low, high, step=None, log=False)`**: 建议一个浮点数。
  - `name` (str): 参数的名称。
  - `low` (float): 搜索范围的下界（包含）。
  - `high` (float): 搜索范围的上界（包含）。
  - `step` (float, 可选): 如果指定，则建议离散的浮点数值。
  - `log` (bool): 若为 `True`，则在对数尺度上采样。这对于学习率 (`learning_rate`) 等参数非常有效，因为它能更均匀地探索 `0.001` 和 `0.01` 之间，以及 `0.01` 和 `0.1` 之间的区域。

- **`trial.suggest_categorical(name, choices)`**: 从一个列表中建议一个类别。
  - `name` (str): 参数的名称。
  - `choices` (list): 包含所有可能选项的列表，例如 `['gbtree', 'dart']`。

- **`trial.suggest_discrete_uniform(name, low, high, q)`**: 建议一个离散均匀分布的值。
  - `name` (str): 参数的名称。
  - `low` (float): 搜索范围的下界（包含）。
  - `high` (float): 搜索范围的上界（包含）。
  - `q` (float): 离散化步长。

- **`trial.suggest_loguniform(name, low, high)`**: 建议一个对数均匀分布的值（已弃用，推荐使用 `suggest_float` 并设置 `log=True`）。

- **`trial.suggest_uniform(name, low, high)`**: 建议一个均匀分布的值（已弃用，推荐使用 `suggest_float`）。
**与 `hyperopt` 分布函数的映射关系：**

| hyperopt | optuna | 含义 |
|----------|--------|------|
| `hp.choice(label, options)` | `trial.suggest_categorical(name, choices)` | 从离散选项中选择一个值，适用于类别型参数 |
| `hp.randint(label, upper)` | `trial.suggest_int(name, 0, upper-1)` | 返回范围 [0, upper-1] 内的随机整数 |
| `hp.uniform(label, low, high)` | `trial.suggest_float(name, low, high)` | 在 [low, high] 范围内均匀采样浮点数 |
| `hp.quniform(label, low, high, q)` | `trial.suggest_float(name, low, high, step=q)` | 在 [low, high] 范围内按步长 q 均匀采样离散值 |
| `hp.loguniform(label, low, high)` | `trial.suggest_float(name, np.exp(low), np.exp(high), log=True)` | 在对数空间上均匀采样，适用于需要探索多个数量级的参数 |
| `hp.qloguniform(label, low, high, q)` | `trial.suggest_float(name, np.exp(low), np.exp(high), log=True, step=q)` | 在对数空间上按步长 q 均匀采样离散值 |
| `hp.normal(label, mu, sigma)` | *无直接对应，可通过自定义采样器实现* | 从正态分布（均值 mu，标准差 sigma）中采样 |
| `hp.lognormal(label, mu, sigma)` | *无直接对应，可通过自定义采样器实现* | 从对数正态分布中采样，适用于非负且有长尾分布的参数 |

### 第3步：创建 Study 并运行优化
在 `optuna` 中，我们首先创建一个 `study` 对象来管理整个优化过程，然后调用其 `optimize` 方法来启动优化。

```python
# 3. 创建 study 对象并运行优化
# direction='minimize' 表示我们的目标是最小化 objective 函数的返回值
study = optuna.create_study(direction='minimize')

# 调用 optimize 方法启动优化
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

**函数说明：**

- **`optuna.create_study()`**: 创建一个 `study` 对象，它是优化任务的控制器。
  - `direction` (str): 优化方向。`'minimize'` (默认) 表示目标是最小化返回值，`'maximize'` 表示目标是最大化返回值。
  - `sampler` (Sampler, 可选): 指定采样算法。默认为 `TPESampler` (即TPE算法)。
  - `pruner` (Pruner, 可选): 指定剪枝器，用于提前终止没有希望的试验。
  - `study_name` (str, 可选): 研究的名称，在存储研究时很有用。
  - `storage` (str 或 None, 可选): 存储研究的数据库URL。

**可用的采样器 (`sampler`) 说明：**

- **`TPESampler`**: 默认采样器，基于树状结构Parzen估计器的贝叶斯优化算法。
  ```python
  # 完整配置示例
  from optuna.samplers import TPESampler
  sampler = TPESampler(
    seed=42,                    # 随机种子
    n_startup_trials=10,        # 初始随机采样的试验数
    multivariate=True,          # 是否使用多变量TPE
    prior_weight=1.0,           # 先验分布的权重
    consider_magic_clip=True,   # 使用魔术剪切来稳定核密度估计
    consider_endpoints=True,    # 在核密度估计中是否考虑端点
    n_ei_candidates=24          # EI最大化中的候选点数量
  )
  ```

- **`RandomSampler`**: 纯随机搜索采样器，类似于scikit-learn的RandomizedSearchCV。
  ```python
  from optuna.samplers import RandomSampler
  sampler = RandomSampler(seed=42)
  ```

- **`CmaEsSampler`**: 使用CMA-ES（协方差矩阵自适应进化策略）算法，特别适合连续参数的优化。
  ```python
  from optuna.samplers import CmaEsSampler
  sampler = CmaEsSampler(
    seed=42,
    x0=None,           # 初始平均向量
    sigma0=0.1,        # 初始步长
    n_startup_trials=1 # 在启动CMA-ES前的随机试验数
  )
  ```

- **`NSGAIISampler`**: 用于多目标优化的非支配排序遗传算法II (NSGA-II)。
  ```python
  from optuna.samplers import NSGAIISampler
  sampler = NSGAIISampler(
    seed=42,
    population_size=50,  # 每代的个体数量
    crossover_prob=0.9,  # 交叉概率
    mutation_prob=None   # 变异概率
  )
  ```

- **`GridSampler`**: 传统的网格搜索采样器，会遍历所有参数组合。
  ```python
  from optuna.samplers import GridSampler
  search_space = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
  }
  sampler = GridSampler(search_space)
  ```

- **`BruteForceSampler`**: 暴力采样器，用于枚举所有可能的离散参数组合。

- **`PartialFixedSampler`**: 部分参数固定的采样器，用于条件搜索空间。

- **`MOTPESampler`**: 多目标树状结构Parzen估计器采样器，用于多目标优化。

### 第4步：查看优化结果
优化完成后，所有结果都保存在 `study` 对象中，可以通过其属性和方法方便地获取。

```python
# 4. 查看优化结果
print("\n" + "="*50)
print("优化完成")
print("="*50)

# study.best_trial 包含了关于最佳试验的所有信息
best_trial = study.best_trial
print(f"最佳损失值 (1 - accuracy): {best_trial.value:.4f}")
print(f"对应的交叉验证准确率: {1 - best_trial.value:.4f}")

# study.best_params 直接返回最佳超参数的字典
print("找到的最佳超参数组合:")
print(study.best_params)

# 可视化优化历史
import optuna.visualization as vis
import matplotlib.pyplot as plt

# 绘制优化历史
vis.plot_optimization_history(study).show()

# 绘制参数重要性
vis.plot_param_importances(study).show()

# 绘制参数之间的相关性
vis.plot_contour(study).show()
```

**`study` 对象结果属性说明：**

- **`study.best_trial`**: 返回一个 `FrozenTrial` 对象，包含了最佳那次试验的全部信息（如参数、值、开始/结束时间等）。
- **`study.best_value`**: 直接返回最佳试验的目标函数值（在这里是最小的 `loss`）。
- **`study.best_params`**: 返回一个字典，包含最佳试验的超参数组合。这是最常用的结果之一，其值的类型已经正确（整数就是整数），无需手动转换。
- **`study.trials`**: 返回一个列表，包含所有已完成的试验对象。
- **`study.trials_dataframe()`**: 将所有试验历史转换为 Pandas DataFrame，非常便于进行深入的分析和可视化。
- **`study.get_trials(deepcopy=True, states=None)`**: 获取满足特定状态的试验。
- **`study.direction`**: 返回优化方向（'minimize' 或 'maximize'）。


## 结论

贝叶斯优化为解决高成本的黑盒优化问题提供了一个强大而严谨的数学框架。它通过构建代理模型（如GPR或TPE）来逼近真实目标函数，并利用采集函数智能地平衡探索与利用，从而高效地找到最优解。GPR为贝叶斯优化提供了坚实的理论基础，而TPE则在处理复杂、高维的实际问题中展现出卓越的性能和扩展性。







