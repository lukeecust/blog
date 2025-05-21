---
title: Sobol 序列：准随机序列生成器及其 Python 实现
author: lukeecust
date: 2025-05-21 02:09:00 +0800
categories: [Deep Learning, Sampling]
tags: [quasi-random sequence]
lang: en
math: true
translation_id: sobol-sequence-generator
permalink: /posts/sobol-sequence-generator/
render_with_liquid: false
---

Random numbers play a crucial role in many fields including cryptography, computer graphics, and statistics. Different applications require different types of random numbers, leading to various types of random number generation. The Sobol sequence is a special type of sequence in the category of quasi-random numbers, widely used in many fields due to its excellent uniform distribution properties.

## Introduction: Importance and Classification of Random Numbers

Before diving into Sobol sequences, let's briefly review the main types of random numbers:

* **Statistically Pseudorandom Numbers:** These are random number sequences that approximate true random sequences in statistical properties (such as uniformity, independence). For example, in a pseudorandom bit stream, the number of 0s and 1s should be roughly equal.
* **Cryptographically Secure Pseudorandom Numbers (CSPRNG):** Beyond meeting the properties of statistical pseudorandom numbers, these require that predicting any part of the sequence from other parts is computationally infeasible. This is crucial for security-related applications like key generation.
* **True Random Numbers:** Generated based on unpredictable physical processes (such as thermal noise, radioactive decay), making samples non-reproducible. Obtaining true random numbers typically comes at a higher cost.

These random number requirements become progressively stricter, with increasing difficulty in generation. Therefore, in practical applications, we need to choose appropriate random number generation methods based on specific requirements.

## Quasi-Random Number Generators (QRNG) and Low-Discrepancy Sequences

In many applications, especially Monte Carlo methods, we need points that are uniformly distributed in the sampling space. **Quasi-Random Number Generators (QRNG)** are designed for this purpose, generating what are known as **Low-Discrepancy Sequences**.

Unlike pseudorandom numbers, low-discrepancy sequences don't aim to "look random" but rather focus on covering the sampling space more uniformly and systematically, avoiding both high clustering of points and large empty regions. Common low-discrepancy sequences include Halton sequences, Faure sequences, Niederreiter sequences, and our focus in this article—**Sobol sequences**.

All random number generation algorithms based on modern CPUs are typically **pseudorandom**, generating sequences through deterministic algorithms that repeat after a very long period. **Quasi-random sequences**, like Sobol sequences, are also deterministic but are designed for low discrepancy, meaning high uniformity.
## Two Important Dimensions of Randomness

From an application perspective, we often need to evaluate random sequences from two dimensions:

1. **Statistical Randomness:** The degree of randomness in a statistical sense, such as uniformity, correlation, and repetition period. This can be assessed through various statistical tests.

2. **Spatial Uniformity:** The distribution characteristics of the sequence in space, especially in multi-dimensional cases. This is typically quantified using discrepancy.

For a point set $P = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ in an $s$-dimensional unit hypercube $[0,1]^s$, its discrepancy $D_N(P)$ can be expressed as:
$$
\begin{equation}
D_N(P) = \sup_{B \in J} \left| \frac{A(B; P)}{N} - \lambda_s(B) \right|
\end{equation}$$

where $J$ is the set of specific shaped subregions in the hypercube, $A(B; P)$ is the number of points falling in subregion $B$, $N$ is the total number of points, and $\lambda_s(B)$ is the volume of the subregion.

Depending on the application scenario, we may focus more on one dimension. For example, statistical randomness is more important in encryption applications, while spatial uniformity may be more critical in numerical integration. Quasi-random sequences (like Sobol sequences) are specifically optimized for spatial uniformity.

## What is a Sobol Sequence?

A Sobol sequence is a series of $n$-dimensional points designed to be more uniformly distributed in the unit hypercube $[0, 1)^n$ than standard pseudo-random sequences.

* **Deterministic:** Points in a Sobol sequence are completely determined for a given dimension and index, unlike pseudo-random numbers that depend on random seeds (although some implementations allow "scrambling" to introduce randomness while maintaining low discrepancy).
* **Low Discrepancy:** This is the core characteristic of Sobol sequences. Discrepancy is a measure of how uniformly points are distributed. Low discrepancy means the point set better avoids large gaps or excessive clustering of points.

**The Concept of Discrepancy**

For a point set $P = \{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ in an $s$-dimensional unit hypercube $[0,1]^s$, its discrepancy $D_N(P)$ is defined as:

$$
\begin{equation}
D_N(P) = \sup_{B \in J} \left| \frac{A(B; P)}{N} - \lambda_s(B) \right|
\end{equation}
$$

where:
* $J$ is the set of all subregions of $[0,1]^s$ with specific shapes (such as axis-aligned subrectangles).
* $A(B; P)$ is the number of points from set $P$ that fall within subregion $B$.
* $N$ is the total number of points in set $P$.
* $\lambda_s(B)$ is the $s$-dimensional volume (or measure) of subregion $B$.

Simply put, discrepancy measures the maximum deviation between the proportion of points in a subregion and the volume of that subregion in the worst case. Point sets with more uniform distribution have lower discrepancy.

The following figure intuitively shows the difference between pseudorandom point sets and low-discrepancy sequence point sets:

![Pseudorandom and Low-discrepancy Sequences](https://lukeecust.github.io/blog/assets/images/2025-05-21-sobol-sequence-generator/discrepancy.png)
_Left: two-dimensional point set composed of pseudorandom numbers; Right: point set from a low-discrepancy sequence (like Sobol sequence), showing more complete and uniform coverage of the space._


## How are Sobol Sequences Generated?

Sobol sequences are generated based on binary arithmetic and a special set of numbers called **direction numbers** or **initialization numbers**. The core idea is related to **Radical Inversion** and **Van der Corput sequences**.

**Radical Inversion and Van der Corput Sequences**

Radical Inversion is a method of mapping an integer $i$ to the interval $[0,1)$. For a base $b$, an integer $i$ can be represented in base $b$ as:
$$\begin{equation}
i = \sum_{l=0}^{M-1} a_l(i) b^l
\end{equation}$$

Its Radical Inversion $\Phi_b(i)$ (in the simplified case where $C$ is the identity matrix, i.e., Van der Corput sequence) is defined as:
$$\begin{equation}
\Phi_b(i) = \sum_{l=0}^{M-1} a_l(i) b^{-l-1}
\end{equation}$$

This effectively mirrors the digits of $i$'s base-$b$ representation from left to right of the decimal point.

For example, the first few terms of the base-2 Van der Corput sequence:
*   $i=1=(1)_2 \implies \Phi_2(1) = (0.1)_2 = 1/2$
*   $i=2=(10)_2 \implies \Phi_2(2) = (0.01)_2 = 1/4$
*   $i=3=(11)_2 \implies \Phi_2(3) = (0.11)_2 = 3/4$
*   $i=4=(100)_2 \implies \Phi_2(4) = (0.001)_2 = 1/8$

Each point in this sequence takes the midpoint of the currently longest uncovered interval, thus ensuring uniform distribution.

**Sobol Sequence Construction**

Each dimension of a Sobol sequence can be viewed as a generalization of a base-2 Van der Corput sequence using different **generating matrices $\mathbf{C}_j$** (corresponding to direction numbers). The $i$-th point $\boldsymbol{X}_i$ of an $n$-dimensional Sobol sequence can be expressed as:
$$\begin{equation}
\boldsymbol{X}_i = \left( \boldsymbol{\Phi}_{2, \mathbf{C}_1}(i), \boldsymbol{\Phi}_{2, \mathbf{C}_2}(i), \ldots, \boldsymbol{\Phi}_{2, \mathbf{C}_n}(i) \right)
\end{equation}$$

where $\boldsymbol{\Phi}_{2, \mathbf{C}_j}(i)$ is the coordinate for dimension $j$, obtained through a series of bitwise XOR operations between the binary representation of integer $i$ and a set of direction numbers (encoded in $\mathbf{C}_j$) for dimension $j$.

Specifically, for the $k$-th point and dimension $j$:
1.  Express $k$ in binary form: $k = (b_m b_{m-1} \dots b_1)_2$
2.  The coordinate $x_{k,j}$ can be expressed as $x_{k,j} = b_1 v_{j,1} \oplus b_2 v_{j,2} \oplus \dots \oplus b_m v_{j,m}$, where $\oplus$ is the XOR operation, and $v_{j,r}$ is the $r$-th direction number for dimension $j$ (itself a binary fraction in $[0,1)$).

Being entirely base-2, Sobol sequences can be efficiently implemented using bitwise operations (like right shifts and XOR), making computation very fast. The choice of primitive polynomials and derived direction numbers is crucial for ensuring the low-discrepancy property of Sobol sequences.

A notable property is that when the number of samples $N$ is a power of 2 (e.g., $N=2^k$), the Sobol sequence will have exactly one point in each base-2 elementary interval of $[0,1)^s$. This means it can generate samples of quality comparable to Stratified Sampling or Latin Hypercube Sampling without requiring the total number of samples to be predetermined.

## Sobol 采样的优缺点
优点：
1.  **优越的均匀性：** 特别是在高维空间，Sobol 序列能比伪随机数更有效地、更均匀地覆盖采样空间。
2.  **更快的收敛速度：** 在数值积分（准蒙特卡洛积分）中，对于 $s$ 维的积分问题，使用 $N$ 个点的标准蒙特卡洛方法的误差收敛速度通常是 $O(N^{-1/2})$。而使用低差异序列（如 Sobol 序列）的准蒙特卡洛方法，其误差收敛速度可以达到$O(N^{-1}(\log N)^s)$或更好。这意味着通常能用更少的样本点达到与 MC 方法相当的精度。
3.  **确定性与可复现性：** 由于序列是确定性生成的（不加扰时），结果是可复现的，这对于调试和比较非常有利。
4.  **高效的参数空间探索：** 适用于需要系统性探索多维参数空间的应用，如超参数优化、灵敏度分析等。
5.  **逐点生成 (Progressive)：** 可以逐点生成，不需要预先知道总样本数 $N$，并且已生成的序列是后续更长序列的前缀，保持了良好的分布特性。这非常适合渐进式采样。

缺点和注意事项：
1.  **初始点的投影可能不佳：** 对于较少的点数（例如，远小于 $2^d$，其中 $d$ 是维度），Sobol 序列在某些低维投影上可能表现出一定的规律性或对齐，看起来不够"随机"。
2.  **方向数的质量：** 序列的质量高度依赖于所使用的方向数。早期的一些方向数集在高维情况下表现可能不佳。现代实现通常使用经过优化的方向数集（例如，由 Joe 和 Kuo 提供的方向数）。
3.  **维度限制：** 虽然理论上 Sobol 序列可以扩展到非常高的维度，但高质量方向数的计算和存储会变得困难。对于极高维度（例如数千维），其相对于标准蒙特卡罗的优势可能会减弱或需要更复杂的加扰技术。通常，Sobol 序列在几十到几百维的问题中表现良好。
4.  **加扰 (Scrambling)：** 为了缓解初始点投影不佳的问题并改善有限样本下的随机性外观，可以对 Sobol 序列进行"加扰"（如随机线性加扰或数字移位）。加扰会引入一定的随机性，但旨在保持低差异特性。


## Sobol 序列与规则网格采样 ( `linspace`) 的对比

一个常见的问题是：既然 Sobol 序列的目标是均匀分布，为什么不直接使用像 `np.linspace` 这样的函数在每个维度上创建等距点，然后组合它们形成一个规则的多维网格呢？

虽然规则网格在低维（如1D或2D）下直观且易于实现均匀覆盖，但在多维空间和许多实际应用场景中，Sobol 序列等低差异序列通常更具优势。主要原因如下：

1.  **维度灾难 (Curse of Dimensionality)：**
    *   **规则网格：** 若在 $d$ 维空间中，每个维度取 $k$ 个点，总点数将是 $k^d$。随着维度 $d$ 的增加，所需点数会呈指数级增长，迅速变得计算上不可行。例如，10 个维度各取 10 个点就需要 $10^{10}$ （一百亿）个样本。
    *   **Sobol 序列：** 可以灵活地生成任意数量 $N$ 的样本点，这些点共同致力于均匀填充 $d$ 维空间，而 $N$ 通常远小于 $k^d$，使其在高维情况下更为实用。

2.  **投影特性与"对齐"伪影：**
    *   **规则网格：** 点严格排列在网格线上，形成高度规则的结构。这种规律性可能导致采样点与被研究函数或现象的特定结构对齐，从而产生系统性偏差。
    *   **Sobol 序列：** 虽然也是确定性的，但其设计旨在最小化"差异性"，使得点在各种子区域（尤其是轴对齐的）中分布更均匀，避免了网格的僵硬结构，并力求在低维投影上也展现良好的分布。

3.  **逐点生成 (Progressive Property)：**
    *   **规则网格：** 通常需要预先确定总点数和网格结构。若需增加样本，往往要重新生成整个更密的网格，原有样本可能无法直接复用。
    *   **Sobol 序列：** 具有逐点生成的特性。可以先生成 $N_1$ 个点，如果需要更高精度，可以继续生成额外的点，形成一个包含 $N_1+N_2$ 个点的序列，该序列的前 $N_1$ 个点与原序列一致，且整个序列仍保持低差异性。这对于渐进式改进和自适应采样非常有利。

4.  **积分收敛性与效率：**
    *   **规则网格 (用于数值积分)：** 虽然某些基于网格的求积法则（如梯形或辛普森法则）对光滑函数有较好的收敛阶，但它们受限于固定的网格结构，且在高维下常数因子可能很差。
    *   **Sobol 序列 (用于准蒙特卡罗积分, QMC)：** QMC 方法的误差收敛速度理论上可达 $O(N^{-1}(\log N)^s)$，通常优于标准蒙特卡罗的 $O(N^{-1/2})$。对于中高维度，QMC 通常比依赖固定网格的确定性求积规则更灵活且高效。

5.  **空间"填充"方式：**
    *   **规则网格：** 像是在空间中规则地铺设"瓷砖"。
    *   **Sobol 序列：** 更像是在空间中"智能地"布置点，以确保覆盖全面且避免空隙和聚集，同时不像网格那样死板。


`linspace` 生成的规则网格主要保证了**单维度上的等距性**，而 Sobol 序列则致力于实现**整个多维空间的低差异性和高均匀度**。因此，在高维问题、需要逐点生成能力、或追求更快积分收敛速度的应用中，Sobol 序列是比规则网格采样更优越的选择。规则网格采样更适用于维度非常低、或需要严格控制各维度采样位置的简单场景。


## Sobol 采样的主要应用领域

Sobol 采样因其优越的特性被广泛应用于多个领域：
1.  **数值积分 (Numerical Integration)：** 这是 Sobol 序列最经典和最主要的应用，即准蒙特卡洛积分。
2.  **金融工程 (Financial Engineering)：** 用于衍生品定价（如期权定价）、风险价值 (VaR) 计算、信用风险模型等。
3.  **计算机图形学 (Computer Graphics)：** 用于全局光照算法（如路径追踪、光线追踪中的采样）、抗锯齿等，以产生更平滑、更真实的图像。
4.  **灵敏度分析 (Sensitivity Analysis)：** 评估模型输出对输入参数变化的敏感程度，Sobol 采样可以有效地探索参数空间。
5.  **优化 (Optimization)：** 作为某些全局优化算法（如基于粒子群的算法或模拟退火）的初始化或搜索策略。
6.  **物理和工程模拟：** 在需要进行大量模拟和参数研究的领域。
7.  **机器学习：** 例如在超参数优化中探索参数组合，以期更有效地找到最优配置。

## Python 中的 Sobol 序列实现

Python 中有多个库提供了 Sobol 序列的实现，其中最常用的是 SciPy 和 PyTorch。

### 使用 SciPy (`scipy.stats.qmc.Sobol`)

SciPy 的 `stats.qmc` (Quasi-Monte Carlo) 模块提供了 `Sobol` 类。

```python
if dimension == 2:
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)  # 调整为2x2布局

    # 普通 Sobol 序列
    axs[0, 0].scatter(samples[:, 0], samples[:, 1], s=20, marker='o', label=f'Sobol (N={num_samples})')
    axs[0, 0].set_title('Standard Sobol Sequence')
    axs[0, 0].set_xlabel('Dimension 1')
    axs[0, 0].set_ylabel('Dimension 2')
    axs[0, 0].set_aspect('equal', adjustable='box')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 伪随机数作为对比
    pseudo_random_samples = np.random.rand(num_samples, dimension)
    axs[0, 1].scatter(pseudo_random_samples[:, 0], pseudo_random_samples[:, 1], s=20, marker='x', color='red',
                      label=f'Pseudo-Random (N={num_samples})')
    axs[0, 1].set_title('Pseudo-Random Samples')
    axs[0, 1].set_xlabel('Dimension 1')
    axs[0, 1].set_ylabel('Dimension 2')
    axs[0, 1].set_aspect('equal', adjustable='box')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 加扰的 Sobol 序列
    axs[1, 0].scatter(samples_scrambled[:, 0], samples_scrambled[:, 1], s=20, marker='s', color='green',
                      label=f'Scrambled Sobol (N={num_samples})')
    axs[1, 0].set_title('Scrambled Sobol Sequence')
    axs[1, 0].set_xlabel('Dimension 1')
    axs[1, 0].set_ylabel('Dimension 2')
    axs[1, 0].set_aspect('equal', adjustable='box')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 拉丁超立方采样 (LHS)
    lhs_engine = qmc.LatinHypercube(d=dimension, seed=42)
    samples_lhs = lhs_engine.random(n=num_samples)
    axs[1, 1].scatter(samples_lhs[:, 0], samples_lhs[:, 1], s=20, marker='P', color='purple',
                      label=f'LHS (N={num_samples})')
    axs[1, 1].set_title('Latin Hypercube Sampling')
    axs[1, 1].set_xlabel('Dimension 1')
    axs[1, 1].set_ylabel('Dimension 2')
    axs[1, 1].set_aspect('equal', adjustable='box')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('Comparison of Sampling Methods (2D)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局适应标题
    plt.savefig('sampling_methods_2d.png', dpi=600)
    plt.show()
```

![采样方法对比](https://lukeecust.github.io/blog/assets/images/2025-05-21-sobol-sequence-generator/sampling_methods_2d.png)

**说明：**
*   `qmc.Sobol(d=dimension, scramble=False)`: 初始化一个 Sobol 序列生成器。`d` 是维度。`scramble=True` 会启用加扰，这通常能改善有限样本的质量，但会失去纯粹的确定性（加扰本身是随机的，但对于固定的种子，加扰后的序列是确定的）。
*   `sobol_engine.random(n=num_samples)`: 生成 `num_samples` 个样本点。每个点是一个 `dimension` 维的向量，其分量在 $[0, 1)$区间内。
*   `sobol_engine.fast_forward(m)`: 可以跳过序列中的前 `m` 个点。
*   `seed`: 当 `scramble=True` 时，`seed` 控制加扰的随机性，以确保可复现性。

### 使用 PyTorch (`torch.quasirandom.SobolEngine`)

PyTorch 也提供了 `SobolEngine` 用于生成 Sobol 序列，这对于在 PyTorch 生态系统中进行工作（例如，在深度学习模型的超参数搜索或基于梯度的期望估计中）非常方便。

```python
import torch
from torch.quasirandom import SobolEngine

# 1. 初始化 SobolEngine
dimension = 2
# scramble=True 进行加扰, seed 用于复现加扰结果
sobol_engine_torch = SobolEngine(dimension=dimension, scramble=False, seed=None)

# 2. 生成样本点
num_samples = 128
# draw 方法返回一个 Tensor
samples_torch = sobol_engine_torch.draw(num_samples) 

print(f"\nGenerated {num_samples} Sobol samples using PyTorch (dimension {dimension}):")
print(samples_torch[:5])

# 3. 使用加扰
sobol_engine_torch_scrambled = SobolEngine(dimension=dimension, scramble=True, seed=42)
samples_torch_scrambled = sobol_engine_torch_scrambled.draw(num_samples)
print("\nScrambled Sobol samples using PyTorch:")
print(samples_torch_scrambled[:5])

```

**说明：**
*   `SobolEngine(dimension=dimension, scramble=False, seed=None)`: 初始化引擎。`dimension` 是维度。`scramble=True` 启用加扰。`seed` 用于在加扰时固定随机数生成器状态。
*   `sobol_engine_torch.draw(num_samples)`: 生成 `num_samples` 个样本点，返回一个 PyTorch `Tensor`。
*   PyTorch 的 `SobolEngine` 最高支持约 1111 维（截至较新版本，具体请查阅官方文档），并且其方向数是经过优化的。

## 不同采样方法的比较

为了更好地理解 Sobol 序列的特性，下表总结了它与其他几种常见采样方法的对比：

| 特性               | 伪随机数 (PRNG)                      | 网格采样 (Grid Sampling)                | 拉丁超立方采样 (LHS)                     | Halton/Hammersley 序列            | Sobol 序列                         |
| :----------------- | :----------------------------------- | :-------------------------------------- | :--------------------------------------- | :----------------------------------- | :--------------------------------- |
| **类型**           | 伪随机 (Pseudo-Random)               | 确定性 / 系统性 (Deterministic/Systematic) | 分层随机 (Stratified Random)             | 准随机 (Quasi-Random)                | 准随机 (Quasi-Random)              |
| **均匀性/覆盖度**  | 可能出现聚集和空隙                     | 规则，但可能在高维下效率低；易产生规律性伪影 | 确保每个一维投影上的分层均匀             | 良好，但低维投影可能不如 Sobol       | 非常好，尤其在高维下                 |
| **差异性**         | 较高                                 | 较低（但结构固定）                        | 中等至较低                               | 低                                   | 非常低                             |
| **收敛速度 (积分)** | $O(N^{-1/2})$                      | 取决于函数，可能较差                     | 通常优于 PRNG，接近 $O(N^{-1})$ (视情况) | $O(N^{-1}(\log N)^s)$ 或更好      | $O(N^{-1}(\log N)^s)$ 或更好       |
| **确定性**         | 否 (依赖种子)                        | 是                                      | 否 (分层内随机)                          | 是                                   | 是 (不加扰时)                      |
| **逐点生成**       | 是                                   | 通常否 (需预定总点数和网格结构)         | 通常否 (需预定总点数)                    | 是                                   | 是                                 |
| **计算成本 (生成)** | 非常低                               | 低                                      | 低至中等                                 | 低                                   | 低 (基于位操作)                    |
| **相关性/模式**    | 低（理论上），但周期有限                | 强烈的规律性，轴对齐模式                 | 避免一维投影的聚集，但高维投影可能仍有结构 | 某些基数选择下初始点可能线性相关     | 初始点投影可能不佳 (可通过加扰改善) |
| **主要优点**       | 实现简单，速度快                       | 简单直观                                | 保证一维投影的良好覆盖，对某些模型有效   | 良好的均匀性，确定性                | 极佳的均匀性，快速收敛，逐点生成     |
| **主要缺点**       | 收敛慢，高维下覆盖不均                  | "维度灾难"，高维下点数剧增，不灵活        | 高维下均匀性可能不如 QMC，非逐点        | 初始点问题，某些基数选择敏感          | 初始点问题，方向数质量依赖          |
| **典型用例**       | 通用随机模拟，游戏，密码学 (CSPRNG)   | 低维参数扫描，可视化                     | 计算机实验设计，不确定性量化，优化     | 数值积分，参数空间探索              | 数值积分，金融，图形学，灵敏度分析   |
| **Python 示例**    | `numpy.random.rand()` `torch.rand()` | `numpy.meshgrid()` `itertools.product()` | `scipy.stats.qmc.LatinHypercube`       | `scipy.stats.qmc.Halton`           | `scipy.stats.qmc.Sobol` `torch.quasirandom.SobolEngine` |

**一些解释：**
*   **网格采样 (Grid Sampling):** 指的是在每个维度上取等间隔的点，然后组合它们。虽然在低维下直观，但在高维下所需的点数会爆炸式增长（维度灾难）。
*   **分层采样 (Stratified Sampling):** 思想是将空间划分为若干不重叠的子区域（层），然后在每个子区域内独立采样。LHS 是分层采样的一种特殊形式。
*   **拉丁超立方采样 (LHS):** 将每个维度划分为 $N$ 个等概率的区间，然后从每个区间中随机抽取一个值，确保在每个维度的每个分层中都有一个样本点，并将这些值随机组合。它保证了一维投影的均匀性。
*   **Halton/Hammersley 序列:** 也是经典的低差异序列，Halton 基于不同素数为基的 Van der Corput 序列，Hammersley 则在其基础上修改第一维。

Sobol 序列通常被认为是综合性能最好的低差异序列之一，尤其是在需要高维均匀采样和快速收敛的准蒙特卡罗积分中。


## 总结


Sobol 序列作为一种经典的低差异序列，通过其确定性的生成方式和优异的均匀分布特性，在多维空间采样方面表现出色。它能够比传统的伪随机数更有效地填充采样空间，从而在数值积分、金融建模、计算机图形学以及机器学习等多个领域中提高计算效率和结果精度。

尽管存在一些如初始点分布和维度限制等问题，但通过现代实现中使用的优化方向数和加扰技术，Sobol 序列仍然是科学计算和工程实践中一个强大且有价值的工具。Python 中的 SciPy 和 PyTorch 等库使得 Sobol 序列的获取和使用变得非常便捷。
