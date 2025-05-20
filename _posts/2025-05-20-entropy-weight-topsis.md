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
    where $x_{ij}$ represents the original value of the $i$-th evaluation object under the $j$-th indicator. The more dispersed the values of an indicator, the greater its role in comprehensive evaluation. If all evaluation values for an indicator are equal, that indicator plays no role in the evaluation and its weight should be 0.

2.  **Data Preprocessing (Forward Transformation and Standardization)**
    To eliminate the impact of different dimensions and inconsistent indicator directions on evaluation results, indicators need to be forward transformed and standardized, converting all indicators into a form where larger values are better and the value range is unified (typically [0, 1]).
    There are generally three types of indicators:
    *   **Positive Indicators (Benefit Type):** Larger is better, such as income, output, scores, satisfaction, etc.
    *   **Negative Indicators (Cost Type):** Smaller is better, such as cost, energy consumption, delay time, failure rate, etc.
    *   **Moderate Indicators (Interval Type):** Indicator values are best when falling within a specific interval or close to a specific value, such as PH value in water (closer to 7 is better), temperature (optimal within a certain range).

    The commonly used method is **Min-Max Normalization**, which accomplishes both forward transformation and standardization simultaneously. The processed matrix is denoted as $Z'$, with elements $z'_{ij}$.


    *   **Positive Indicators:**
        $$
        \begin{equation}
        z'_{i j}=\frac{x_{i j}-x_j^{\min }}{x_j^{\max }-x_j^{\min }}
        \end{equation}
        $$
        where $x_j^{\min }$ and $x_j^{\max }$ are the minimum and maximum values of the $j$-th indicator across all objects.

    *   **Negative Indicators:**
        $$
        \begin{equation}
        z'_{i j}=\frac{x_j^{\max }-x_{i j}}{x_j^{\max }-x_j^{\min }}
        \end{equation}
        $$

    *   **Moderate Indicators:**
        *   If the optimal value is a specific point $x_{best_j}$, use the following formula:
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
        
        After processing, all $z'\_\{ij\}$ values fall within the $[0,1]$ interval and are all positively oriented. If $\{z'\_\{\{ij\}\}=0\}$ occurs, the subsequent calculation of information entropy will involve $\ln(p\_{ij})$. To avoid the undefined case when $p\_{ij}=0$ (since $\ln(p\_{ij})$ is not defined), it is usually stipulated that when $p\_{ij}=0$, its contribution to the entropy calculation, $p\_{ij} \ln p\_{ij}$, is set to $0$. Another approach is to add a very small positive number $\epsilon$ (e.g., 0.0001) to all $z'_{ij}$, i.e., ${z''}\_{\{ij\}} = z'\_\{ij\} + \epsilon$, before proceeding with subsequent calculations. However, this shift may slightly alter the relative differences in the original data, so the first convention is generally preferred.
3.  **Calculate the proportion $p_{ij}$ of the $i$-th object under the $j$-th indicator**  
    For the normalized matrix $Z'$, calculate the contribution or proportion of the $i$-th evaluation object on the $j$-th indicator:
    $$
    \begin{equation}
    p_{ij} = \frac{z'_{ij}}{\sum_{k=1}^{n} z'_{kj}}
    \end{equation}
    $$

4.  **Calculate the entropy value $e_j$ of the $j$-th indicator**  
    $$
    \begin{equation}
    e_j = -k \sum_{i=1}^{n} (p_{ij} \ln p_{ij})
    \end{equation}
    $$
    where the constant $k = \frac{1}{\ln n}$ and $n$ is the number of evaluation objects. The purpose of $k$ is to normalize the entropy value $e_j$ to the $[0,1]$ interval.

5.  **Calculate the degree of divergence (information redundancy) $d_j$ of the $j$-th indicator**  
    The degree of divergence $d_j$ is obtained by subtracting the information entropy $e_j$ from 1:
    $$
    \begin{equation}
    d_j = 1 - e_j
    \end{equation}
    $$
    The larger $d_j$ is, the more information the $j$-th indicator contains, and the more important it is in the evaluation, thus it should be assigned a larger weight.

6.  **Calculate the weight $w_j$ of the $j$-th indicator**  
    Normalize the degree of divergence of each indicator to obtain the final weights:
    $$
    \begin{equation}
    w_j = \frac{d_j}{\sum_{k=1}^{m} d_k}
    \end{equation}
    $$
    where $m$ is the number of indicators. This ensures that the sum of all indicator weights is 1, i.e., $\sum_{j=1}^{m} w_j = 1$.

7.  **(Optional) Preliminary comprehensive evaluation based on the entropy weight method**  
    If only the entropy weight method is used for comprehensive evaluation (without combining with TOPSIS or other methods), the weighted comprehensive score $F_i$ for each evaluation object can be directly calculated:
    $$
    \begin{equation}
    F_i = \sum_{j=1}^{m} w_j z'_{ij}
    \end{equation}
    $$
    However, it is more common to use the weights $w_j$ calculated by the entropy weight method as input for subsequent multi-attribute decision-making methods (such as TOPSIS). In this case, a **weighted normalized decision matrix $V$** is constructed, with elements $v_{ij} = w_j z'_{ij}$. This matrix $V$ is an important starting point for the TOPSIS method.
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

## TOPSIS Method

The TOPSIS method, short for "Technique for Order Preference by Similarity to Ideal Solution," is a widely used Multi-Attribute Decision Making (MADM) approach.

**Core Concept:**  
The core idea of TOPSIS is to rank alternatives based on their relative closeness to the "ideal solution" and "negative ideal solution."
*   **Positive Ideal Solution ($V^+$):** A hypothetical best alternative, where each indicator achieves the optimal value among all alternatives.
*   **Negative Ideal Solution ($V^-$):** A hypothetical worst alternative, where each indicator takes the least favorable value among all alternatives.

By calculating the (weighted) Euclidean distance from each alternative to the positive and negative ideal solutions, alternatives that are closer to the positive ideal and farther from the negative ideal are considered superior.

**Integration with the Entropy Weight Method:**  
1.  **Data Normalization:** The first step of TOPSIS is usually data normalization and positive transformation, which can directly use the standardized matrix $Z'$ obtained in Step 2 of the Entropy Weight Method.
2.  **Indicator Weights:** TOPSIS requires indicator weights when calculating distances or constructing the weighted decision matrix. These can be directly taken from the weight vector $W = [w_1, w_2, \dots, w_m]$ calculated in Step 6 of the Entropy Weight Method.

**Calculation Steps:**

1.  **Construct the Weighted Normalized Matrix $V$**  
    Using the weights $w_j$ obtained from the entropy weight method and the normalized data $z'_{ij}$, construct the weighted normalized matrix $V$:
    $$
    \begin{equation}
    v_{ij} = w_j z'_{ij}
    \end{equation}
    $$
    The matrix form is as follows:
    $$
    V=\left[\begin{array}{cccc}
    v_{11} & v_{12} & \cdots & v_{1 m} \\
    v_{21} & v_{22} & \cdots & v_{2 m} \\
    \vdots & \vdots & \ddots & \vdots \\
    v_{n 1} & v_{n 2} & \cdots & v_{n m}
    \end{array}\right]
    $$

2.  **Determine the Positive Ideal Solution $V^+$ and Negative Ideal Solution $V^-$**  
    Since all indicators have been positively oriented (the larger, the better), the positive ideal solution is composed of the maximum value in each column of the weighted normalized matrix $V$, and the negative ideal solution is composed of the minimum value in each column.
    *   **Positive Ideal Solution $V^+$:**
        $$
        \begin{equation}
        V^+ = (V_1^+, V_2^+, \dots, V_m^+) = (\max_i v_{i1}, \max_i v_{i2}, \dots, \max_i v_{im})
        \end{equation}
        $$
    *   **Negative Ideal Solution $V^-$:**
        $$
        \begin{equation}
        V^- = (V_1^-, V_2^-, \dots, V_m^-) = (\min_i v_{i1}, \min_i v_{i2}, \dots, \min_i v_{im})
        \end{equation}
        $$

3.  **Calculate the Distance from Each Alternative to the Positive and Negative Ideal Solutions**  
    The Euclidean distance is usually used to calculate the distance from the $i$-th alternative to the positive ideal solution $V^+$ ($D_i^+$) and to the negative ideal solution $V^-$ ($D_i^-$):
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

4.  **Calculate the Relative Closeness $C_i$ (also called the Comprehensive Evaluation Value or Closeness Coefficient) for Each Alternative**  
    The relative closeness $C_i$ of the $i$-th alternative is defined as:
    $$
    \begin{equation}
    C_i = \frac{D_i^{-}}{D_i^{+} + D_i^{-}}
    \end{equation}
    $$
    The value of $C_i$ ranges from $0$ to $1$. The larger the $C_i$, the closer the alternative $i$ is to the positive ideal solution and the farther it is from the negative ideal solution, indicating a better overall evaluation. By ranking all alternatives according to $C_i$, the order of preference can be determined.


## Combined Application of Entropy Weight Method and TOPSIS

The integration of the Entropy Weight Method (EWM) with the TOPSIS method demonstrates unique advantages:

*   **Advantages of EWM:** Provides an objective approach for calculating indicator weights, avoiding biases that may arise from subjective weighting, and making the evaluation results more scientific.
*   **Advantages of TOPSIS:** Establishes a systematic multi-alternative evaluation framework, enabling scientific ranking of alternatives through distance measurement from the ideal solution.

In practical applications, the EWM is first used to objectively determine the weights of each indicator. These weights are then used as input parameters for the TOPSIS method, which systematically analyzes and ranks the evaluation objects. This combination not only ensures the objectivity of weight determination but also guarantees the reliability of the final evaluation results.

**Different Approaches in the Literature:**

1.  **Most Common Approach:**
    *   First, use the entropy weight method to calculate the weights $w_j$.
    *   Then, use $w\_j$ and the normalized data $z'\_\{ij\}$ to construct the weighted normalized matrix $V = (w\_j z'\_\{ij\})$.
    *   Determine the positive and negative ideal solutions $V^+, V^-$ based on $V$.
    *   Calculate the distance from each alternative to $V^+$ and $V^-$ (at this point, the distance formula no longer explicitly includes $w_j$, as it is already incorporated in $v_{ij}$).
    *   This is the clearest and most widely used approach.

2.  **Weights Reflected in Distance Calculation:**
    *   First, normalize the original data to obtain $Z'$.
    *   Determine the positive ideal solution $Z'^+$ and negative ideal solution $Z'^-$ based on the normalized matrix $Z'$.
    *   When calculating the distance from each alternative to $Z'^+$ and $Z'^-$, introduce the weights $w_j$ obtained from the entropy weight method:
        $$
        \begin{equation}
        \begin{aligned}
        D_i^{+} & =\sqrt{\sum_{j=1}^m w_j (z'_{ij} - Z_j'^{+})^2} \\
        D_i^{-} & =\sqrt{\sum_{j=1}^m w_j (z'_{ij} - Z_j'^{-})^2}
        \end{aligned}
        \end{equation}
        $$

        This method is mathematically different from the first approach (for example, if $w\_j$ is a proportion, then in the first approach $V\_j^+$ is $\max(w\_j z'\_\{ij\})$, while in the second approach $Z\_j'^+$ is $\max(z'\_\{ij\})$). The first approach is generally considered more standard, as it incorporates the effect of weights when constructing the ideal solutions. However, this weighted Euclidean distance is also a valid method.

3.  **Weights Used in Both Steps:**
    *   First, construct $V=(\{w\_j\} z^{\prime}\_\{\{ij\}\})$, then determine $V^+, V^-$, and subsequently use $D\_i^{+} =\sqrt{\sum\_{j=1}^m w\_j (v\_{ij} - V\_j^{+})^2}$ to calculate the distance. This approach essentially applies the weights twice (once in $v\_{ij}$, and once in the$w\_j$ of the distance formula), which may lead to over-amplification or unreasonable interpretation of the weights, unless there is a specific theoretical basis (for example, the external $w\_j$ could be a function of $w\_j$, such as $w\_j^2$). In general, this double weighting is not recommended.



## Summary

The entropy weight-TOPSIS-based multi-objective optimization decision analysis method objectively determines indicator weights using the entropy weight method, and then ranks alternatives with the TOPSIS method. This is a practical and effective decision support tool, capable of handling complex decision problems with multiple evaluation indicators, and producing results that are relatively objective and scientific.

**Advantages:**
*   **High objectivity:** The entropy weight method relies entirely on the data itself, eliminating subjective factors.
*   **Good comprehensiveness:** The TOPSIS method considers the distance to both the optimal and worst alternatives, fully reflecting the comprehensive performance of each alternative.
*   **Simple principle, easy to understand and implement:** The calculation process is straightforward and suitable for programming and application.

**Limitations and Considerations:**
*   **Sensitivity of the entropy weight method:** When the sample size is small or data variability is low, the weights calculated by the entropy weight method may be unstable or lack discrimination.
*   **Ranking stability of the TOPSIS method:** Like some other multi-attribute decision-making methods, TOPSIS may experience rank reversal when alternatives are added or removed (although its stability is relatively good).
*   **Assumption of indicator independence:** Traditional entropy weight and TOPSIS methods usually assume that indicators are independent, without fully considering correlations between indicators.
*   **Sensitivity to outliers:** The min-max normalization process can be affected by extreme data points.

**Outlook:**
Future research may explore further integration of the entropy weight-TOPSIS method with other approaches, such as:
*   Combining subjective weighting methods (such as AHP) with the entropy weight method to form composite weights, balancing expert experience and data objectivity.
*   Improving the TOPSIS method, for example by considering different distance metrics or introducing prospect theory, to adapt to more complex decision scenarios.
*   Studying methods to handle indicator correlations, such as integrating principal component analysis (PCA) or grey relational analysis (GRA).



