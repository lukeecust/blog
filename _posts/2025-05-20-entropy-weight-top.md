---
title: Multi-Objective Decision Analysis Based on Entropy Weight Method and TOPSIS
author: lukeecust
date: 2025-05-20 02:09:00 +0800
categories: [Multi-Objective Optimization, Decision Analysis]
lang: en
math: true
translation_id: entropy-weight-topsis
permalink: posts/entropy-weight-topsis/
render_with_liquid: false
---

In real-world decision-making problems, it is often necessary to consider multiple conflicting or interrelated objectives simultaneously, known as multi-objective optimization decision problems. How to scientifically and objectively evaluate different alternatives and select the optimal solution is a core issue in decision analysis. The Entropy Weight Method (EWM), as an objective weighting method, can determine indicator weights based on data variability, effectively avoiding subjective interference. The Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) is a classical ranking method that orders alternatives by calculating their relative proximity to the best and worst solutions.

By combining the Entropy Weight Method with TOPSIS, we first use EWM to determine objective weights for evaluation indicators, then apply these weights in TOPSIS for comprehensive evaluation and ranking of alternatives. This combined approach leverages both the objective weighting advantage of EWM and the systematic multi-alternative ranking capability of TOPSIS, finding wide applications in project evaluation, performance assessment, risk analysis, and other fields.


## Entropy Weight Method

The main purpose of the Entropy Weight Method is to objectively assign weights to indicator systems based on the characteristics of the data itself.

**Basic Principles:**
The Entropy Weight Method is an objective weighting method based on information entropy. Information entropy measures the degree of uncertainty or disorder in a system. In multi-indicator evaluation:
- If an indicator's observed values show large differences (high degree of variation), the indicator contains more useful information and should be assigned a larger weight, with lower information entropy
- If an indicator's observed values show small differences (low degree of variation), the indicator has weak discriminating power for decision-making and should be assigned a smaller weight, with higher information entropy

The Entropy Weight Method determines weights by calculating the information entropy of each indicator. The smaller the information entropy of an indicator, the greater its utility value and weight; conversely, the smaller. This method relies entirely on the objective attributes of the data.


**Calculation Steps:**

1.  **Data Preparation**
    Assume there are $n$ evaluation objects (samples) and $m$ evaluation indicators, forming the original data matrix $X$:
    $$
    \begin{equation}
    X=\left[\begin{array}{cccc}
    x_{11} & x_{12} & \cdots & x_{1 m} \\
    x_{21} & x_{22} & \cdots & x_{2 m} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n 1} & x_{n 2} & \cdots & x_{n m}
    \end{array}\right]
    \end{equation}
    $$
    where $x_{ij}$ represents the original value of the $i$-th evaluation object under the $j$-th indicator. For a given indicator, the greater the dispersion of its values, the more significant its role in comprehensive evaluation. If all evaluation values for an indicator are equal, that indicator has no effect in the evaluation, and its weight should be 0.

2.  **Data Preprocessing (Forward Transformation and Normalization)**
    To eliminate the influence of different dimensions and inconsistent indicator directions on evaluation results, all indicators need to be forward transformed and normalized, converting them into a form where larger values are better and the value range is unified (usually [0, 1]).
    There are generally three types of indicators:
    *   **Positive Indicators (Benefit Type):** Larger is better, such as income, output, scores, satisfaction, etc.
    *   **Negative Indicators (Cost Type):** Smaller is better, such as cost, energy consumption, delay time, failure rate, etc.
    *   **Moderate Indicators (Interval Type):** Indicator values are best when falling within a specific interval or close to a specific value, such as PH in water (closer to 7 is better), temperature (optimal within a certain range).

    A commonly used method is **Min-Max Normalization**, which completes both forward transformation and normalization. The processed matrix is denoted as $Z'$, with elements $z'_{ij}$.

    *   **For Positive Indicators:**
        $$
        \begin{equation}
        z'_{i j}=\frac{x_{i j}-x_j^{\min }}{x_j^{\max }-x_j^{\min }}
        \end{equation}
        $$
        where $x_j^{\min }$ and $x_j^{\max }$ are the minimum and maximum values of the $j$-th indicator among all objects.

    *   **For Negative Indicators:**
        $$
        \begin{equation}
        z'_{i j}=\frac{x_j^{\max }-x_{i j}}{x_j^{\max }-x_j^{\min }}
        \end{equation}
        $$

    *   **For Moderate Indicators:**
        *   If the optimal value is a point $x_{best_j}$ (e.g., PH=7), use the following formula:
        $$
        \begin{equation}
        z'_{i j}=1-\frac{\left|x_{i j}-x_{best_j}\right|}{\max_k \left(\left|x_{k j}-x_{best_j}\right|\right)}
        \end{equation}
        $$
        *   If the optimal value is an interval $[a_j, b_j]$, then:
        $$
        \begin{equation}
        z'_{i j}= \begin{cases}
        1-\frac{a_j-x_{i j}}{\max \left(a_j-x_j^{\min }, x_j^{\max }-b_j\right)} & , x_{i j}<a_j \\
        1 & , a_j \leq x_{i j} \leq b_j \\
        1-\frac{x_{i j}-b_j}{\max \left(a_j-x_j^{\min }, x_j^{\max }-b_j\right)} & , x_{i j}>b_j
        \end{cases}
        \end{equation}
        $$
        After processing, all values of $z'_{ij}$ fall within the interval $[0,1]$ and are forward transformed.

    极差法标准化后，矩阵 $Z'$ 中的值均在 $[0,1]$ 区间。若出现 $z'_{ij}=0$ 的情况，在后续计算信息熵时，会涉及到 $\ln(p_{ij})$。为避免 $p_{ij}=0$ 导致 $\ln(p_{ij})$ 无意义，通常约定当 $p_{ij}=0$ 时，其在熵值计算中的贡献项 $p_{ij} \ln p_{ij} = 0$。
    另一种处理方式是对所有 $z'_{ij}$ 加上一个极小的正数 $\epsilon$ (例如0.0001)，即 $z''_{ij} = z'_{ij} + \epsilon$，然后再进行后续计算。但这种平移可能会轻微改变原始数据的相对差异，通常优先采用前一种约定。

3.  **计算第 $j$ 项指标下第 $i$ 个对象所占比重 $p_{ij}$**
    对于标准化后的矩阵 $Z'$ ，计算第 $i$ 个评价对象在第 $j$ 个指标上的贡献度或比重：
    $$
    \begin{equation}
    p_{ij} = \frac{z'_{ij}}{\sum_{k=1}^{n} z'_{kj}}
    \end{equation}
    $$

4.  **计算第 $j$ 项指标的熵值 $e_j$**
    $$
    \begin{equation}
    e_j = -k \sum_{i=1}^{n} (p_{ij} \ln p_{ij})
    \end{equation}
    $$
    其中，常数 $k = \frac{1}{\ln n}$，$n$ 为评价对象的数量。$k$ 的作用是使得熵值 $e_j$ 规范化到 $[0,1]$ 区间。

5.  **计算第 $j$ 项指标的差异程度（信息冗余度） $d_j$**
    指标的差异程度 $d_j$ 用 $1$ 减去其信息熵 $e_j$ 得到：
    $$
    \begin{equation}
    d_j = 1 - e_j
    \end{equation}
    $$
    $d_j$ 越大，表示第 $j$ 个指标的信息越多，其对于评价的重要性也越大，应赋予更大的权重。

6.  **计算第 $j$ 项指标的权重 $w_j$**
    将各指标的差异程度进行归一化处理，得到各指标的最终权重：
    $$
    \begin{equation}
    w_j = \frac{d_j}{\sum_{k=1}^{m} d_k}
    \end{equation}
    $$
    其中 $m$ 为指标的数量。确保所有指标权重之和为1，即 $\sum_{j=1}^{m} w_j = 1$。

7.  **（可选）基于熵权法的初步综合评价**
    如果仅使用熵权法进行综合评价（不结合TOPSIS等其他方法），可以直接计算每个评价对象的加权综合得分 $F_i$：
    $$
    \begin{equation}
    F_i = \sum_{j=1}^{m} w_j z'_{ij}
    \end{equation}
    $$
    然而，更常见的做法是将熵权法计算得到的权重 $w_j$ 作为后续多属性决策方法（如TOPSIS）的输入。此时，会构建**加权标准化决策矩阵 $V$**，其元素 $v_{ij} = w_j z'_{ij}$。这个矩阵 $V$ 是TOPSIS方法的重要起点。
    $$
    \begin{equation}
    V = \left[\begin{array}{cccc}
    w_1 z'_{11} & w_2 z'_{12} & \cdots & w_m z'_{1m} \\
    w_1 z'_{21} & w_2 z'_{22} & \cdots & w_m z'_{2m} \\
    \vdots & \vdots & \ddots & \vdots \\
    w_1 z'_{n1} & w_2 z'_{n2} & \cdots & w_m z'_{nm}
    \end{array}\right]
    \end{equation}
    $$

## TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

TOPSIS法，全称为“逼近理想解排序法”，是一种常用的多属性决策（MADM）方法。

**核心思想：**
TOPSIS法的核心思想是基于评价对象与“理想解”和“负理想解”的相对接近程度来进行排序。
*   **正理想解 ($V^+$)：** 一个虚拟的最佳方案，其每个指标值都达到所有备选方案中的最优水平。
*   **负理想解 ($V^-$)：** 一个虚拟的最劣方案，其每个指标值都达到所有备选方案中的最差水平。

通过计算每个评价对象到正理想解和负理想解的（加权）欧氏距离，如果一个评价对象越接近正理想解，同时越远离负理想解，则该对象越优。

**与熵权法的结合点：**
1.  **数据标准化：** TOPSIS的第一步通常也是数据正向化和标准化，可以直接采用熵权法步骤2得到的标准化矩阵 $Z'$。
2.  **指标权重：** TOPSIS在计算距离或构建加权决策矩阵时需要各指标的权重，这可以直接采用熵权法步骤6计算得到的权重向量 $W = [w_1, w_2, \dots, w_m]$。

**计算步骤：**

1.  **构建加权标准化矩阵 $V$**
    使用熵权法得到的权重 $w_j$ 和标准化后的数据 $z'_{ij}$，构建加权标准化矩阵 $V$：
    $$
    \begin{equation}
    v_{ij} = w_j z'_{ij}
    \end{equation}
    $$
    矩阵形式如下：
    $$
    V=\left[\begin{array}{cccc}
    v_{11} & v_{12} & \cdots & v_{1 m} \\
    v_{21} & v_{22} & \cdots & v_{2 m} \\
    \vdots & \vdots & \ddots & \vdots \\
    v_{n 1} & v_{n 2} & \cdots & v_{n m}
    \end{array}\right]
    $$

2.  **确定正理想解 $V^+$ 和负理想解 $V^-$**
    由于所有指标都已正向化（越大越好），正理想解由加权标准化矩阵 $V$ 中每列的最大值构成，负理想解由每列的最小值构成。
    *   **正理想解 $V^+$：**
        $$
        \begin{equation}
        V^+ = (V_1^+, V_2^+, \dots, V_m^+) = (\max_i v_{i1}, \max_i v_{i2}, \dots, \max_i v_{im})
        \end{equation}
        $$
    *   **负理想解 $V^-$：**
        $$
        \begin{equation}
        V^- = (V_1^-, V_2^-, \dots, V_m^-) = (\min_i v_{i1}, \min_i v_{i2}, \dots, \min_i v_{im})
        \end{equation}
        $$

3.  **计算各评价对象到正、负理想解的距离**
    通常使用欧氏距离来计算第 $i$ 个评价对象与正理想解 $V^+$ 的距离 $D_i^+$ 和与负理想解 $V^-$ 的距离 $D_i^-$：
    $$
    \begin{equation}
    D_i^{+} = \sqrt{\sum_{j=1}^m (v_{ij} - V_j^{+})^2}
    \end{equation}
    $$
    $$
    \begin{equation}
    D_i^{-} = \sqrt{\sum_{j=1}^m (v_{ij} - V_j^{-})^2}
    \end{equation}
    $$

4.  **计算各评价对象的相对接近度 $C_i$ (也称综合评价值或贴近度)**
    第 $i$ 个评价对象的相对接近度 $C_i$ 定义为：
    $$
    \begin{equation}
    C_i = \frac{D_i^{-}}{D_i^{+} + D_i^{-}}
    \end{equation}
    $$
    $C_i$ 的取值范围为 $[0, 1]$。$C_i$ 越大，表示评价对象 $i$ 越接近正理想解且越远离负理想解，因此其综合评价越优。根据 $C_i$ 的值对所有评价对象进行排序，即可得到方案的优劣次序。


## 熵权法 + TOPSIS 结合应用

熵权法与TOPSIS法的结合应用展现出独特的优势：

*   **熵权法优势**：提供客观的指标权重计算方法，避免主观赋权可能带来的偏差，使评价结果更具科学性。
*   **TOPSIS法优势**：构建了系统化的多方案评价框架，通过与理想解的距离度量实现方案的科学排序。

在实际应用中，首先运用熵权法计算得到各指标的客观权重，然后将这些权重作为TOPSIS法的输入参数，对评价对象进行系统性分析和排序。这种结合不仅保证了权重确定的客观性，也确保了最终评价结果的可靠性。

**论文中存在的不同做法：**

1.  **最常见做法：**
    *   先用熵权法计算出权重 $w_j$。
    *   然后用 $w_j$ 和标准化数据 $z'_{ij}$ 构建加权标准化矩阵 $V = (w_j z'_{ij})$。
    *   基于 $V$ 确定正负理想解 $V^+, V^-$。
    *   计算各方案到 $V^+, V^-$ 的距离（此时距离公式中不再显式出现 $w_j$，因为它已包含在 $v_{ij}$ 中）。
    *   这是逻辑最清晰、应用最广泛的方式。

2.  **权重在距离计算中体现：**
    *   先对原始数据进行标准化处理得到 $Z'$。
    *   基于标准化矩阵 $Z'$ 确定正理想解 $Z'^+$ 和负理想解 $Z'^-$。
    *   在计算各方案到 $Z'^+, Z'^-$ 的距离时，引入熵权法得到的权重 $w_j$：
        $$
        \begin{aligned}
        D_i^{+} & =\sqrt{\sum_{j=1}^m w_j (z'_{ij} - Z_j'^{+})^2} \\
        D_i^{-} & =\sqrt{\sum_{j=1}^m w_j (z'_{ij} - Z_j'^{-})^2}
        \end{aligned}
        $$
        这种方法在数学上与第一种方法中的距离计算有所不同（例如，如果 $w_j$ 是比例，那么第一种方法的 $V_j^+$ 是 $\max(w_j z'_{ij})$，而第二种方法的 $Z_j'^+$ 是 $\max(z'_{ij})$）。通常认为第一种方法更为标准，因为它在构建理想解时就考虑了权重的影响。但这种加权欧氏距离也是一种有效的方式。

3.  **权重在两处均使用：**
    *   “以上两处均使用权重，即使用加权标准化矩阵计算正负理想解，并在计算对象的正负理想解距离时使用权重。”
    *   如果这里的含义是：先构建 $V=(w_j z'_{ij})$，然后确定 $V^+, V^-$，之后计算距离时再用 $D_i^{+} =\sqrt{\sum_{j=1}^m w_j (v_{ij} - V_j^{+})^2}$。这种做法相当于权重被应用了两次（一次在 $v_{ij}$ 中，一次在距离公式的 $w_j$ 中），可能会导致权重的过度放大或不合理解释，除非有特殊的理论依据（例如，外部的 $w_j$ 可能是 $w_j$ 的某个函数，如 $w_j^2$）。一般不推荐这种重复加权。



## 总结

基于熵权法-TOPSIS的多目标优化决策分析方法，通过熵权法客观确定指标权重，再结合TOPSIS法对方案进行排序，是一种实用且有效的决策支持工具。它能够较好地处理具有多个评价指标的复杂决策问题，结果相对客观、科学。

**优势：**
*   **客观性强：** 熵权法赋权完全依赖于数据本身，排除了主观因素。
*   **综合性好：** TOPSIS法同时考虑了与最优和最劣方案的距离，能全面反映方案的综合表现。
*   **原理简单，易于理解和实现：** 计算过程相对直观，便于编程实现和应用。

**局限性与注意事项：**
*   **熵权法的敏感性：** 当样本数据量较少或数据波动性不大时，熵权法计算的权重可能不够稳定或区分度不高。
*   **TOPSIS法的排序稳定性：** 与某些多属性决策方法类似，TOPSIS在增加或删除备选方案时，可能会出现排序逆转的现象（尽管其稳定性相对较好）。
*   **指标独立性假设：** 传统熵权法和TOPSIS法通常假设指标之间是相互独立的，未充分考虑指标间的相关性。
*   **对极端值敏感：** 标准化过程中的最大最小值易受极端数据点影响。

**展望：**
未来可以探索熵权法-TOPSIS与其他方法的进一步融合，例如：
*   结合主观赋权法（如AHP）与熵权法，形成组合权重，兼顾专家经验与数据客观性。
*   改进TOPSIS法，如考虑不同距离度量、引入前景理论等，以适应更复杂的决策场景。
*   研究处理指标相关性的方法，如结合主成分分析（PCA）或灰色关联分析（GRA）等。



