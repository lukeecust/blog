---
title: Takagi-Sugeno (T-S) Fuzzy Model
author: lukeecust
date: 2025-05-24 14:09:00 +0800
categories: [Machine Learning, Model, Fuzzy]
tags: [model]
lang: en
math: true
translation_id: takagi-sugeno-fuzzy-model
permalink: /posts/takagi-sugeno-fuzzy-model/
render_with_liquid: false
---


In many real-world systems, the behavior of objects or processes is often nonlinear, complex, and difficult to describe with precise mathematical models. Traditional modeling methods (such as first-principles modeling based on physical laws) may be too complex or impossible to establish. Data-driven modeling methods (such as black-box models) can fit data but may lack interpretability or make system analysis difficult (especially for stability analysis and controller design).

Fuzzy Logic Systems (FLS) provide a modeling and control method based on human language and experiential knowledge to handle uncertainty and fuzziness. Among the many types of fuzzy logic systems, the **Takagi-Sugeno (T-S) fuzzy model** (also known as T-S fuzzy system or TSK fuzzy system) has been widely applied in system modeling, identification, and control due to its unique structure and good mathematical properties.

The T-S fuzzy model was proposed by Japanese scholars Toshio Takagi and Michio Sugeno in 1985. It combines the advantages of fuzzy logic and the analyzability of mathematical models, making it particularly suitable for modeling nonlinear dynamic systems and model-based control design.

## Principles of T-S Fuzzy Model

The core idea of the T-S fuzzy model is to use a set of fuzzy rules to describe a complex nonlinear system. Unlike traditional Mamdani fuzzy models (where the consequent is a fuzzy set), the **consequent of a T-S fuzzy model is a clear mathematical function**, typically a linear combination of input variables.

### Rule Structure

A typical T-S fuzzy model consists of $M$ fuzzy rules, each having the following form:

**Rule $i$**: **IF** $x_1$ is $A_{i1}$ **AND** $x_2$ is $A_{i2}$ **AND** ... **AND** $x_n$ is $A_{in}$
**THEN** $y_i = f_i(x_1, x_2, ..., x_n)$

Where:
*   $i = 1, 2, ..., M$ represents the $i$-th rule.
*   $x = [x_1, x_2, ..., x_n]^T$ is the vector of input variables.
*   $A_{ij}$ is the fuzzy set describing input variable $x_j$.
*   $y_i$ is the output of the $i$-th rule.
*   $f_i(x)$ is the consequent function of the $i$-th rule, typically a linear function of the input variables:
    $y_i = p_{i0} + p_{i1}x_1 + p_{i2}x_2 + ... + p_{in}x_n$

**About connectives in rules:**
Standard T-S fuzzy rule antecedents typically use **AND (conjunction)** to connect judgments of input conditions (e.g., $x_1$ is $A_{i1}$ AND $x_2$ is $A_{i2}$ ...). This means the activation degree of the rule depends on the degree to which all antecedent conditions are **simultaneously satisfied**. If there's a need to express an **OR (disjunction)** relationship (e.g., "if temperature is high or pressure is high, then..."), instead of directly using OR within **one** rule antecedent, this logic is typically implemented by constructing **multiple rules**. For example, "IF Temperature is High OR Pressure is High THEN ..." can be decomposed into two T-S rules:
*   Rule 1: IF Temperature is High THEN ...
*   Rule 2: IF Pressure is High THEN ...

The total output of the system is a weighted average of these rule outputs, indirectly achieving the effect of OR. The standard T-S model structure (especially state-space models used for control analysis) typically relies on the AND structure in the antecedent to define local regions.

###  Fuzzification and Premise Activation Calculation

For a given input $x = [x_1, x_2, ..., x_n]^T$, first calculate the membership degree $\mu_{A_{ij}}(x_j) \in [0, 1]$ of each input variable $x_j$ to each fuzzy set $A_{ij}$.

**Common Membership Function Types:**
Choosing appropriate membership function types significantly impacts model performance. Common types include:

1.  **Triangular Membership Function:**
    *   Simple definition, determined by three parameters (left vertex, peak, right vertex).
    *   Advantages: Simple calculation, intuitive.
    *   Disadvantages: Not differentiable at vertices, unsuitable for gradient-based optimization algorithms; not smooth enough.
    *   Applications: Scenarios with clear concept boundaries, or where computational efficiency is required.

2.  **Trapezoidal Membership Function:**
    *   Determined by four parameters, including a flat top region.
    *   Advantages: Simple calculation, can represent values that completely belong to a fuzzy set within a certain range.
    *   Disadvantages: Not differentiable at corners, unsuitable for gradient-based optimization algorithms; not smooth enough.
    *   Applications: Scenarios where concept boundaries have some "tolerance" or flat regions.

3.  **Gaussian Membership Function:**
    *   $\mu(x) = e^{-\frac{(x-c)^2}{2\sigma^2}}$: Determined by center and standard deviation parameters, bell-shaped curve.
    *   Advantages: Smooth, differentiable everywhere, very suitable for gradient-based learning algorithms (like neural network training). Enables smooth transitions.
    *   Disadvantages: Slightly higher computation than triangular/trapezoidal functions; parameters less intuitive than vertices.

4.  **Generalized Bell Membership Function:**
    *   $$\mu(x) = \frac{1}{1 + \left\lvert\frac{x-c}{a}\right\rvert^{2b}}$$: Determined by three parameters, providing more flexible shape adjustment than Gaussian functions.
    *   Advantages: Smooth, differentiable, flexible shape.
    *   Disadvantages: Slightly higher computation, parameters less intuitive than center and width.
    *   Applications: Similar to Gaussian functions, providing more flexibility.

There is no absolutely "best" membership function type. The choice depends on:
*   **System characteristics and prior knowledge:** If certain inputs are known to behave stably within specific ranges (flat top), trapezoidal might be suitable. For smooth transitions and gradient optimization, Gaussian or generalized bell functions are better choices.
*   **Modeling objectives:** If interpretability is emphasized, triangular or trapezoidal may be more intuitive. If fitting accuracy and learning ability are emphasized, Gaussian or generalized bell functions typically perform better.
*   **Computational resources:** Triangular/trapezoidal calculations are fastest.

Calculate the premise activation degree (firing strength) for each rule. The premise activation degree represents the applicability of that rule under the current input. For standard T-S models where antecedent conditions are connected by **AND**, there are two common antecedent conjunction (AND) operators (T-norms):

1.  **Product operator:**
    $w_i = \mu_{A_{i1}}(x_1) \times \mu_{A_{i2}}(x_2) \times ... \times \mu_{A_{in}}(x_n)$
    *   Advantages: Smooth, differentiable everywhere, suitable for gradient-based learning algorithms. The premise activation degree reflects the "joint probability" or degree of "simultaneous satisfaction" of all conditions.
    *   Disadvantages: If any membership degree is zero, the activation degree of that rule becomes zero, even if other membership degrees are high.

2.  **Minimum operator:**
    $w_i = \min(\mu_{A_{i1}}(x_1), \mu_{A_{i2}}(x_2), ..., \mu_{A_{in}}(x_n))$
    *   Advantages: Intuitive, indicates that the rule's activation is limited by the least satisfied condition.
    *   Disadvantages: Not differentiable when multiple membership degrees are equal and minimal, potentially affecting gradient-based learning.

**Applications and Pros/Cons Summary:** The product operator typically performs better in data-driven modeling and learning because it's smoother and aids optimization. The minimum operator better aligns with the intuitive understanding of "AND" in human language (barrel principle) and may be more common in rule-based expert systems, but is less popular than the product operator in modeling scenarios requiring optimization.

###  Fuzzy Inference and Output Composition (Weighted Average)

The inference process of a T-S fuzzy model directly produces a clear output value. Each rule's premise activation degree $w_i$ is associated with that rule's consequent function $y_i = f_i(x)$.

The total output of the system is obtained by weighted averaging of all rule consequent outputs. The weights are the corresponding rule premise activation degrees. Weight normalization is typically performed:

Normalized activation degree: $$\bar{w}_i = \frac{w_i}{\sum_{j=1}^{M} w_j}$$

where $$\sum_{j=1}^{M} w_j > 0$$ is required.

The final output $y$ of the model is the weighted average of all rule consequent outputs:

$$\begin{equation}
y = \sum_{i=1}^{M} \bar{w}_i y_i = \sum_{i=1}^{M} \frac{w_i}{\sum_{j=1}^{M} w_j} f_i(x)
\end{equation}$$

If the consequent function is linear $f_i(x) = p_{i0} + p_{i1}x_1 + ... + p_{in}x_n$, then the total output is:

$$\begin{equation}
y = \sum_{i=1}^{M} \bar{w}_i (p_{i0} + p_{i1}x_1 + ... + p_{in}x_n)
\end{equation}$$

**About Weighting Methods and Centroid Method:**
*   **Weighting in T-S Models:** T-S models directly use **normalized premise activation degrees** as weights for the **clear consequent output of each rule**. This is a **direct mathematical calculation process**.
*   **Centroid Method in Mamdani Models:** In Mamdani models, the output of each rule is a **fuzzy set**. After all rule output fuzzy sets are aggregated (combined) into a total output fuzzy set, a **defuzzification** step is needed to convert this total fuzzy set into a single crisp value. The centroid method (Centroid of Area, COA) is one of the most common defuzzification methods in Mamdani models, calculating the **geometric center of the total output fuzzy set area**.
*   **Is the Centroid Method Applicable to T-S Models?** **No.** The output composition process (weighted average) in T-S models directly produces a crisp output value $y$. There is no **total output fuzzy set** as in Mamdani that needs centroid calculation. Therefore, the centroid method's role in Mamdani models (converting the total output fuzzy set to a crisp value) is unnecessary in T-S models, and the weighted average is the final output calculation method for T-S models.

## Advantages and Characteristics of T-S Fuzzy Models

Compared to other fuzzy models or traditional modeling methods, T-S fuzzy models have the following significant advantages and characteristics:

1.  **Local Linear Approximation Capability**: T-S models can effectively approximate and represent complex nonlinear functions or system dynamics through weighted combinations of local linear models. This rule-based structure gives them good **function approximation** and **interpolation** capabilities.
2.  **Convenient for Analysis and Control Design**: This is one of the most important advantages of T-S models, **directly benefiting from their clear consequent function form**. Since the consequent is a mathematical function, especially a linear function, mature linear system theory tools can be used to analyze the properties of T-S fuzzy models. For example, by transforming T-S fuzzy systems into Linear Matrix Inequality (LMI) problems, stability analysis and controller design based on Parallel Distributed Compensation (PDC) can be conveniently performed.
3.  **Model Parameter Identifiability**: T-S fuzzy model parameters can be identified from system input-output data through various learning algorithms. In particular, parameters of consequent linear functions can be directly solved using classical methods such as least squares, while parameters of antecedent membership functions can be adjusted through gradient descent or optimization algorithms. This parameter learnability makes T-S models a powerful tool for data-driven modeling, adaptively capturing system dynamics.
4.  **Structured and Modular**: T-S fuzzy models decompose complex nonlinear problems into multiple relatively simple local linear subproblems, with each fuzzy rule corresponding to a local model. This modular structure allows the model to be flexibly expanded (adding new rules) or simplified (merging or deleting redundant rules). Meanwhile, individual rules can be modified or optimized relatively independently, facilitating model maintenance and updates.
5.  **Grey-box Modeling Potential**: T-S fuzzy models combine knowledge-based rule descriptions (antecedent part) and data-driven parameter learning (consequent part), thus can be viewed as a "grey-box" model, intermediate between transparent "white-box" models (like physical models) and opaque "grey-box" models (like neural networks). This allows T-S models to simultaneously leverage domain expert knowledge and system measurement data, achieving good fitting accuracy while maintaining a certain level of model interpretability.

##  T-S Fuzzy Model Modeling Process

Building a T-S fuzzy model typically includes the following steps:

1.  **Determine Input and Output Variables**: Select key input and output variables based on the system or process to be modeled.
2.  **Determine Rule Number and Perform Fuzzy Partitioning**:
    *   Determine the number of rules based on system complexity, input variable ranges, and data distribution.
    *   Perform fuzzy partitioning of the input variable space, i.e., define fuzzy sets ($A_{ij}$) for each input variable in each rule's antecedent, and select appropriate membership function types (such as triangular, Gaussian) and their parameters. This step is a key component of building the **rule base**. This is typically done by combining expert experience and/or clustering analysis (such as Fuzzy C-Means clustering, FCM) to determine the number, type, and initial position of membership functions.
3.  **Determine Consequent Function Structure and Parameters**:
    *   Linear functions are typically chosen as consequents.
    *   Consequent function parameters ($p_{i0}, p_{i1}, ..., p_{in}$) are typically identified from input-output data through learning algorithms. Common methods include:
        *   **Clustering-based and Least Squares Method**: Use clustering algorithms (like FCM) to cluster input-output data pairs $(x_k, y_k)$. Each cluster center can be used to initialize the antecedent membership functions of a rule. Then, for data points in each cluster, assuming they locally conform to a linear model $y_k \approx p_{i0} + p_{i1}x_{k1} + ... + p_{in}x_{kn}$, use least squares to solve for the $i$-th rule consequent parameters $$\mathbf{p}_i = [p_{i0}, ..., p_{in}]^T$$. This is a two-stage learning process.
        *   **Optimization-based Methods**: Use the error between model output and actual output as the optimization objective, employing gradient descent, heuristic algorithms, or other methods to simultaneously adjust membership function parameters and consequent function parameters to minimize the error.
        *   **Hybrid Learning Algorithms**: For example, **ANFIS** (Adaptive Neuro-Fuzzy Inference System) is a common network structure implementing T-S models that combines the advantages of neural networks and fuzzy systems. It typically uses a hybrid learning algorithm: least squares for quickly determining consequent linear parameters, and gradient descent for adjusting antecedent membership function parameters.
4.  **Model Validation and Optimization**: Evaluate model performance (such as prediction accuracy) using independent test data. If performance is unsatisfactory, it may be necessary to adjust the number of rules, membership function shapes, parameter identification methods, etc., for iterative optimization.

### **Dynamic T-S Fuzzy Models**

Especially in the control field, it's often necessary to establish dynamic system models. A discrete-time dynamic T-S fuzzy model can be represented as:

**Rule $i$**: **IF** $z_t$ is $A_i$
**THEN** $\mathbf{x}_{t+1} = A_i \mathbf{x}_t + B_i \mathbf{u}_t$
$y_t = C_i \mathbf{x}_t + D_i \mathbf{u}_t$

Where:
*   $\mathbf{x}_t$ is the state vector of the system at time $t$.
*   $\mathbf{u}_t$ is the control input vector of the system at time $t$.
*   $y_t$ is the output vector of the system at time $t$.
*   $z_t$ is the antecedent variable vector, typically including the system's state variables $\mathbf{x}_t$ and/or input variables $\mathbf{u}_t$.
*   $A_i, B_i, C_i, D_i$ are system matrices associated with the $i$-th rule, defining the local linear dynamics in that fuzzy region.

The total dynamic equation of the system is obtained by weighted averaging of the consequents of all rules:

$$\begin{equation}\mathbf{x}_{t+1} = \sum_{i=1}^{M} \bar{w}_i(z_t) (A_i \mathbf{x}_t + B_i \mathbf{u}_t)\end{equation}$$
$$\begin{equation}y_t = \sum_{i=1}^{M} \bar{w}_i(z_t) (C_i \mathbf{x}_t + D_i \mathbf{u}_t)\end{equation}$$

Where $\bar{w}_i(z_t)$ is the normalized activation degree calculated based on the antecedent variable $z_t$. This dynamic T-S model forms the basis for model-based fuzzy control (such as PDC).

##  Applications of T-S Fuzzy Models

The unique structure and advantages of T-S fuzzy models enable their wide application in numerous fields:

1.  **System Modeling & Identification**:
    *   Used to establish dynamic models of complex nonlinear systems, such as industrial processes, biological systems, robot dynamics, etc.
    *   Identify system parameters from input-output data through learning algorithms, achieving **system identification**.
2.  **Fuzzy Control**:
    *   **Model-based Control Design**: This is one of the most important applications of T-S fuzzy models. Through establishing a dynamic T-S fuzzy model of the system, controllers can be designed based on this model. The most typical method is **Parallel Distributed Compensation (PDC)**. PDC controllers have the same fuzzy antecedents as the T-S model, but the consequents are linear functions of controller outputs (typically state feedback or output feedback). For example, for the above dynamic T-S model, a state feedback PDC controller rule form is:
  
        **Rule $i$**: **IF** $$z_t$$ is $$A_i$$ **THEN** $$\mathbf{u}_t = K_i \mathbf{x}_t$$

        Then the total control law is: $$\mathbf{u}_t = \sum_{i=1}^{M} \bar{w}_i(z_t) K_i \mathbf{x}_t$$

        Substituting this control law into the dynamic T-S model yields the fuzzy dynamic equation of the closed-loop system. Using Lyapunov stability theory, the stability problem of the closed-loop system can be transformed into a feasibility problem of solving a set of **Linear Matrix Inequalities (LMIs)**. If there exists a common positive definite matrix $P$ and a set of controller gain matrices $K_i$ such that certain LMI conditions are satisfied, then global asymptotic stability (or local stability) of the closed-loop system can be guaranteed. This LMI-based analysis and design method is a core advantage of T-S fuzzy control.
    *   **Nonlinear System Control**: Widely applied in robot control (such as inverted pendulums, robotic arms, where T-S models can describe the dynamics around different equilibrium points), aircraft control, vehicle control, motor control, process control, and other fields, handling system nonlinearities and uncertainties.
3.  **Prediction & Forecasting**:
    *   Used for time series prediction, such as financial data, energy consumption, traffic flow, etc.
    *   Utilizing historical data to establish nonlinear fuzzy mapping relationships between inputs and outputs for prediction.
4.  **Signal Processing**:
    *   Used for signal filtering, pattern recognition, **fault diagnosis**, etc.
5.  **Other Fields**:
    *   Also applied in data mining, decision support, artificial intelligence, and other fields.

##  Limitations of T-S Fuzzy Models

Despite its many advantages, T-S models also have some limitations:

1.  **Curse of Dimensionality**: When the number of input variables increases or the number of fuzzy sets for each input variable increases, if fully connected rules are adopted (every fuzzy set of each input variable combined with all fuzzy sets of other input variables), the number of rules grows exponentially. This leads to a dramatic increase in model parameters, greater training difficulty, increased computation, and potential issues of incomplete rule coverage or overfitting.
2.  **No Systematic Method for Determining Rule Number and Membership Functions**: The selection of rule numbers and the type, shape, and initial parameters of membership functions often rely on expert experience or trial and error, lacking systematic theoretical guidance. This to some extent affects the automation and optimality of the modeling process.
3.  **Complexity of Parameter Learning**: Although parameters are learnable, when the model scale is large, the parameter learning process may become complex and time-consuming, and prone to local optima.
4.  **Limitations of Local Linear Assumption**: The model is based on local linear approximation. For systems with highly nonlinear or discontinuous behavior, a large number of rules may be needed to achieve satisfactory modeling accuracy, exacerbating the curse of dimensionality problem.
5.  **Interpretability Decreases with Scale**: Although individual rules are easy to understand, when the number of rules is huge, the interpretability of the entire model deteriorates.

##  Conclusion

The T-S fuzzy model is a powerful tool for nonlinear system modeling. By combining fuzzy logic antecedent reasoning with mathematical function consequents, it achieves effective approximation of complex nonlinear systems. Its core advantage lies in the mathematical function form of the consequent (especially linear form), which greatly facilitates model analysis (particularly stability analysis, often implemented through LMIs) and model-based controller design (especially the PDC method), making it a powerful weapon for handling nonlinear system modeling, identification, and control problems. Despite challenges such as the "curse of dimensionality" and heuristic parameter determination, T-S fuzzy models and their related analysis and design methods still occupy an important position in theoretical research and engineering practice, representing an important research direction in the fields of intelligent control and data-driven modeling.