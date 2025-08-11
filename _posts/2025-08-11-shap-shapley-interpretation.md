---
title: SHAP Interpretability Analysis Based on Shapley Values
author: lukeecust
date: 2025-08-11 15:09:00 +0800
categories: [Machine Learning, Model Interpretation]
tags: [Feature Importance]
lang: en
math: true
translation_id: shap-shapley-interpretation
permalink: /posts/shap-shapley-interpretation/
render_with_liquid: false
---

With the widespread application of machine learning models, model interpretability has become a key issue. Traditional feature importance methods such as Permutation Importance and gradient-based methods, while simple and intuitive, have theoretical flaws and practical limitations. SHAP (SHapley Additive exPlanations) introduces the concept of Shapley values from game theory, providing a solid theoretical foundation and unified framework for model interpretation.

## Theoretical Foundation

### Shapley Values: From Game Theory to Feature Attribution

Shapley values originated in cooperative game theory to solve fair allocation problems. Consider a cooperative game with $n$ participants, where the cooperation benefit of any subset $S \subseteq N = \{1,2,...,n\}$ is given by the characteristic function $v: 2^N \rightarrow \mathbb{R}$. The Shapley value for participant $i$ is defined as their average marginal contribution across all possible coalitions:

$$\begin{equation}
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}[v(S \cup \{i\}) - v(S)]
\end{equation}$$

The intuitive understanding of this formula is: consider all possible orders in which participants can join, calculate the marginal contribution of participant $$i$$ when they join in each order, then take the average. The weight $$\frac{|S|!(n-|S|-1)!}{n!}$$ represents the probability that set $$S$$ appears before participant $$i$$.

### From Shapley Values to SHAP: Application in Machine Learning

Applying Shapley values to machine learning model interpretation requires establishing the following mapping:
- **Participants** → **Features**: Each feature is viewed as a participant
- **Coalitions** → **Feature subsets**: Combinations of different features
- **Benefit function** → **Model prediction**: Model output corresponding to feature subsets

For a prediction model $f: \mathbb{R}^M \rightarrow \mathbb{R}$ and input sample $x \in \mathbb{R}^M$, the SHAP value for feature $i$ is defined as:

$$\begin{equation}
\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(M-|S|-1)!}{M!}[v(S \cup \{i\}) - v(S)]
\end{equation}$$

where $F = \{1,2,...,M\}$ is the feature set. The key is how to define the value function $v(S)$. SHAP uses conditional expectation:

$$\begin{equation}v(S) = E[f(X)|X_S = x_S] = \int f(x_S, X_{\bar{S}})p(X_{\bar{S}}|X_S = x_S)dX_{\bar{S}}\end{equation}$$

This definition ensures that when we observe the values of feature subset $S$ as $x_S$, the value function reflects the expected prediction of the model.

### Additive Feature Attribution Model and SHAP's Unified Framework

SHAP unifies various interpretation methods under an additive feature attribution framework. The explanation model $g$ is defined as:

$$\begin{equation}g(z') = \phi_0 + \sum_{i=1}^M \phi_i z'_i\end{equation}$$

where $z' \in \{0,1\}^M$ represents simplified features (indicating whether a feature is observed), $\phi_0 = E[f(X)]$ is the baseline value (expected prediction when no features are observed), and $\phi_i$ is the attribution value for feature $i$.

### Theoretical Properties and Advantages of SHAP

SHAP values satisfy the following key properties, which ensure the reasonableness and consistency of the explanation:

1. **Local Accuracy**: $f(x) = \phi_0 + \sum_{i=1}^M \phi_i$
   
   The model's actual prediction equals the baseline value plus the sum of all SHAP values, ensuring the completeness of the explanation.

2. **Missingness**: If feature $i$ does not affect the prediction, then $\phi_i = 0$
   
   Irrelevant features have zero contribution, avoiding false attribution.

3. **Consistency**: If model $f'$ relies more on feature $i$ compared to $f$, then $\phi_i(f', x) \geq \phi_i(f, x)$
   
   When a model relies more on a certain feature, the importance of that feature will not decrease, which is a key property many other methods lack.

**Theorem (Uniqueness)**: SHAP values are the only additive feature attribution method that simultaneously satisfies the above three properties.

This uniqueness theorem is the theoretical cornerstone of the SHAP method, guaranteeing that SHAP provides the only reasonable explanation scheme under the given axiomatic system.

## Comparison with Other Methods

### Comparison with Permutation Importance

Permutation Importance evaluates feature importance by randomly shuffling feature values:

$$\begin{equation}PI_i = E[\mathcal{L}(y, f(X_{-i}^{perm}, X_i))] - E[\mathcal{L}(y, f(X))]\end{equation}$$

where $X_{-i}^{perm}$ represents the data after the $i$-th feature has been randomly permuted.

**Key Differences**:
- **Granularity**: Permutation importance provides global average importance, while SHAP can provide local explanations for each sample
- **Interaction Effects**: Permutation importance attributes feature interactions to single features, while SHAP can explicitly separate main effects and interaction effects
- **Theoretical Guarantees**: SHAP has a rigorous axiomatic foundation, while permutation importance lacks theoretical guarantees
- **Computational Stability**: For correlated features, permutation importance may produce misleading results, while SHAP handles correlations through conditional expectation

### Differences from Partial Derivatives and Correlation Coefficients

**Partial Derivatives**: $\frac{\partial f}{\partial x_i}$ measure local sensitivity of features

- Only considers local linear approximation, ignoring non-linear effects
- Not applicable for categorical features or discrete models
- Cannot handle interaction effects between features

**Correlation Coefficients**: $\rho(x_i, y)$ measure linear correlation

- Only captures linear relationships, ignoring non-linear patterns
- Cannot distinguish between causal relationships and correlations
- Does not consider conditional effects of other features

**Advantages of SHAP**:
- Considers all possible feature combinations, capturing complete non-linear relationships
- Based on marginal contributions, closer to causal explanation
- Provides additive decomposition, facilitating understanding of overall predictions

## Choosing and Comparing SHAP Explainers

Different SHAP explainers are optimized for different types of models and computational needs. Choosing the appropriate explainer is crucial for obtaining accurate and efficient explanations.

| Explainer | Applicable Models | Computation Property | Time Complexity | Core Assumption | Interaction Support | Main Advantages | Limitations |
|--------|----------|----------|------------|----------|----------|----------|---------|
| **TreeExplainer** | Tree models (XGBoost/RF/LightGBM) | Exact/Approximate | $O(TLD^2)$ | Conditional expectation/Path dependency | Native support | Fast, accurate, industry standard | Limited to tree models |
| **KernelExplainer** | Model-agnostic | Approximate | $O(2^M \cdot N)$ | Feature independence | Indirect | Strong universality | Computationally expensive |
| **LinearExplainer** | Linear/GLM | Exact | $O(M^2)$ | Independent/Gaussian | Not supported | Closed-form solution, fast | Strong assumptions |
| **DeepExplainer** | Deep networks | Approximate | $O(P \cdot N)$ | Based on DeepLIFT | Not supported | Suitable for deep learning | Numerical stability issues |
| **GradientExplainer** | Differentiable models | Approximate | $O(N \cdot G)$ | Integrated gradients | Not supported | Theoretically elegant | Computation intensive |

## Core Algorithm Principles

### KernelSHAP Algorithm

KernelSHAP is a model-agnostic method that approximates SHAP values by solving a weighted linear regression. Its core idea is to perform local linear approximation in simplified feature space.

Optimization objective:
$$\begin{equation}
\min_{g} \sum_{z' \subseteq x'} \pi_x(z')[f(h_x(z')) - g(z')]^2
\end{equation}$$

where the weight function is designed as:
$$\begin{equation}
\pi_x(z') = \frac{M-1}{\binom{M}{|z'|} \cdot |z'| \cdot (M - |z'|)}
\end{equation}$$

This weight function design ensures that the solution to the linear regression is exactly the Shapley values. The mapping function $h_x: \{0,1\}^M \rightarrow \mathbb{R}^M$ is defined as: when $z'_i = 1$, take $x_i$, otherwise take the expected value of the background distribution.

**Algorithm Process**:
1. Generate coalition samples $$\{z'_k\}_{k=1}^K$$, prioritizing sampling coalitions with sizes close to 0 or M
2. Calculate the weight for each coalition $\pi_x(z'_k)$
3. Obtain complete inputs through the mapping function $h_x(z'_k)$, calculate model outputs $f(h_x(z'_k))$
4. Solve the weighted least squares problem to get SHAP values

### TreeSHAP Algorithm

TreeSHAP is specifically designed for tree models, utilizing the characteristics of tree structures to achieve accurate and efficient computation. For tree ensemble models, TreeSHAP can calculate SHAP values exactly in polynomial time.

Core recursive relationship:
$$\begin{equation}
v(S) = \sum_{l \in L} b_l \cdot p(l|x_S)
\end{equation}$$

where $$L$$ is the set of leaf nodes, $$b_l$$ is the prediction value of leaf node $$l$$, and $$p(l|x_S)$$ is the probability of reaching leaf node $$l$$ given feature subset $$x_S$$.

TreeSHAP avoids exponential enumeration through dynamic programming, reducing complexity to $O(TLD^2)$, where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum depth.

**Two Conditional Expectation Modes**:
- **Interventional**: Assumes feature independence, $v(S) = E_{X_{\bar{S}} \sim p(X_{\bar{S}})}[f(x_S, X_{\bar{S}})]$
- **Tree Path-Dependent**: Preserves tree path dependency, $$v(S) = E_{X_{\bar{S}} \sim p(X_{\bar{S}}|x_S)}[f(x_S, X_{\bar{S}})]$$

The former is closer to causal explanation, while the latter preserves the inherent structure of tree models.

## Empirical Analysis Implementation

Assuming you have a trained model and data, here's how to conduct a complete SHAP analysis.

### Environment Preparation and Data Loading

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Use example data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assume we have a trained XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
```

### Global Interpretation Analysis

#### Global Quantification of Feature Importance

```python
# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Calculate global feature importance
global_importance = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    'std_shap': np.std(shap_values, axis=0)
}).sort_values('mean_abs_shap', ascending=False)

print("Global feature importance (based on mean absolute SHAP values):")
print(global_importance.head(10))
```

Mean absolute SHAP values quantify the average impact of each feature on predictions. Unlike traditional impurity-based feature importance, SHAP values consider interaction effects between features and have additivity—the sum of all SHAP values equals the difference between the model prediction and the baseline value.

#### Beeswarm Plot: Understanding Feature Impact Distribution

```python
# Beeswarm plot showing the complete distribution of SHAP values
shap.summary_plot(shap_values, X_test, plot_type="violin")
```

The Beeswarm plot (also called summary plot) is one of SHAP's most informative visualizations. Each point represents a sample:
- **Vertical axis**: Features sorted by importance
- **Horizontal axis**: SHAP values, representing the feature's contribution to the prediction
- **Color**: Actual feature value (red indicates high value, blue indicates low value)

Through this plot, you can observe:
1. Overall feature importance (horizontal dispersion of points)
2. Relationship between feature values and direction of impact (relationship between color and horizontal coordinate)
3. Non-linear relationships (if there are vertical color gradient changes)

#### Bar Plot: Concise Importance Ranking

```python
# Bar plot showing mean absolute SHAP values
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

The bar plot provides the most concise view of global feature importance. The length of each bar represents the mean absolute SHAP value of that feature, intuitively showing which features are most important for model predictions. This representation is particularly suitable for presenting the main drivers of the model to non-technical audiences.

#### Dependence Plot: Exploring Non-linear Feature Effects

```python
# Select the most important features for dependence plot analysis
top_features = global_importance['feature'].head(3)

for feature in top_features:
    # Automatically select the best interaction feature
    shap.dependence_plot(feature, shap_values, X_test, interaction_index="auto")
```

Dependence plots reveal the relationship between individual features and their SHAP values, helping to understand:
- **Main effects**: How feature values affect predictions (through the main trend line)
- **Interaction effects**: How other features modulate that impact (through color-coded scatter points)
- **Non-linear patterns**: Identifying threshold effects or saturation effects

The automatic selection of interaction features is based on the feature most strongly correlated with the SHAP values of the main feature, which often reveals meaningful feature interactions.

### Local Interpretation Analysis

#### Waterfall Plot: Decision Decomposition for a Single Sample

```python
# Select a sample with high probability of positive class
sample_idx = np.where(model.predict_proba(X_test)[:, 1] > 0.9)[0][0]

# Create waterfall plot
shap.waterfall_plot(
    shap.Explanation(values=shap_values[sample_idx],
                     base_values=explainer.expected_value,
                     data=X_test.iloc[sample_idx],
                     feature_names=X_test.columns.tolist()),
    max_display=15
)
```

The waterfall plot shows the cumulative process from the baseline value (average prediction of all training samples) to the final prediction value. Each feature's contribution is sorted by magnitude, making the most important factors immediately apparent. Red bars represent features that increase the prediction value, while blue bars represent features that decrease it. This visualization is particularly suitable for explaining the logic behind individual decisions.

#### Force Plot: Push-Pull Analysis of Predictions

```python
# Force plot visualization
shap.force_plot(
    explainer.expected_value, 
    shap_values[sample_idx], 
    X_test.iloc[sample_idx],
    matplotlib=True
)
```

The force plot provides another intuitive perspective on local explanation. It views prediction as a balance of forces from different features:
- **Baseline**: Expected output of the model (average of all training samples)
- **Red forces**: Features that increase the prediction value
- **Blue forces**: Features that decrease the prediction value
- **Final position**: Actual prediction value

The advantage of the force plot is that it compactly displays the contributions of all features while maintaining an intuitive representation of additivity.

#### Decision Plot: Multi-sample Comparison

```python
# Select multiple samples for comparison
sample_indices = [0, 10, 20, 30, 40]
shap.decision_plot(explainer.expected_value, 
                   shap_values[sample_indices], 
                   X_test.iloc[sample_indices])
```

The decision plot shows the path from baseline value to final prediction for multiple samples. Each line represents a sample, allowing observation of:
- **Common patterns**: Decision paths followed by most samples
- **Outliers**: Special cases that deviate from the mainstream path
- **Key divergence points**: Locations where different prediction results fork

This visualization is particularly useful for understanding the model's decision boundaries and identifying edge cases.

### Feature Interaction Effect Analysis

```python
# Calculate SHAP interaction values (only applicable to tree models)
shap_interaction_values = explainer.shap_interaction_values(X_test[:50])  # Use subset to save computation time

# Extract main diagonal (main effects) and off-diagonal (interaction effects)
main_effects = np.diagonal(shap_interaction_values, axis1=1, axis2=2)
total_interactions = np.sum(np.abs(shap_interaction_values), axis=(1, 2)) - np.sum(np.abs(main_effects), axis=1)

# Calculate interaction intensity matrix
interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
np.fill_diagonal(interaction_matrix, 0)  # Remove self-interactions

# Visualize interaction matrix
plt.figure(figsize=(12, 10))
mask = interaction_matrix < np.percentile(interaction_matrix, 95)  # Only show strong interactions
interaction_matrix_masked = np.ma.masked_where(mask, interaction_matrix)

im = plt.imshow(interaction_matrix_masked, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, label='Mean Absolute Interaction Intensity')
plt.title('Feature Interaction Heatmap (Top 5%)')

# Annotate strongest interactions
top_n = 5
indices = np.unravel_index(np.argsort(interaction_matrix.ravel())[-top_n:], interaction_matrix.shape)
for i, j in zip(indices[0], indices[1]):
    if i < j:  # Avoid duplicates
        plt.annotate(f'{i}-{j}', (j, i), color='blue', fontweight='bold', ha='center', va='center')

plt.xlabel('Feature Index')
plt.ylabel('Feature Index')
plt.tight_layout()
plt.show()

# Output strongest interaction pairs
print("\nStrongest feature interaction pairs:")
interaction_pairs = []
for i in range(len(X.columns)):
    for j in range(i+1, len(X.columns)):
        interaction_pairs.append((X.columns[i], X.columns[j], interaction_matrix[i, j]))
        
interaction_pairs_df = pd.DataFrame(interaction_pairs, columns=['Feature1', 'Feature2', 'Interaction'])
print(interaction_pairs_df.nlargest(5, 'Interaction'))
```

Feature interaction analysis reveals synergistic effects in the model. Strong interactions indicate that the joint effect of two features is not equal to the sum of their individual effects, which is very important in understanding complex patterns and building better feature engineering.

### SHAP-based Feature Selection

```python
# Use SHAP values for feature selection
feature_importance = np.abs(shap_values).mean(axis=0)
importance_threshold = np.percentile(feature_importance, 50)  # Select top 50% features

selected_features = X_test.columns[feature_importance > importance_threshold].tolist()
print(f"Selected {len(selected_features)} features (out of {len(X_test.columns)})")

# Verify selection effect
from sklearn.metrics import roc_auc_score

# Using all features
model_full = xgb.XGBClassifier(n_estimators=100, random_state=42)
model_full.fit(X_train, y_train)
auc_full = roc_auc_score(y_test, model_full.predict_proba(X_test)[:, 1])

# Using selected features
model_selected = xgb.XGBClassifier(n_estimators=100, random_state=42)
model_selected.fit(X_train[selected_features], y_train)
auc_selected = roc_auc_score(y_test, model_selected.predict_proba(X_test[selected_features])[:, 1])

print(f"\nPerformance comparison:")
print(f"All features AUC: {auc_full:.4f}")
print(f"Selected features AUC: {auc_selected:.4f}")
print(f"Performance retention rate: {auc_selected/auc_full*100:.1f}%")
```

SHAP values provide a principled method for feature selection. Compared to traditional methods, SHAP-based feature selection considers the actual contribution of features rather than merely correlations, identifying features that are important in specific contexts.

## Theoretical Deepening

### SHAP and Causal Inference

Although SHAP is primarily an associative explanation method, under specific conditions it can approximate causal effects:

$$\begin{equation}
\text{Causal effect} \approx \phi_i \quad \text{when features satisfy conditional independence}
\end{equation}$$

Using interventional TreeSHAP (setting `feature_perturbation='interventional'`) can partially alleviate the bias brought by feature correlations, making explanations closer to causal meaning.

### Computational Complexity and Approximation Algorithms

The complexity of exact SHAP value calculation is exponential $O(2^M)$, and various approximation algorithms are used in practice:

| Method | Complexity | Approximation Quality | Applicable Scenarios |
|------|--------|----------|----------|
| Monte Carlo sampling | $O(K \cdot T_{eval})$ | Depends on sample size K | Universal but slow |
| Linear approximation (KernelSHAP) | $O(K \cdot T_{eval})$ | Local linear assumption | Model-agnostic |
| Path integral (TreeSHAP) | $O(TLD^2)$ | Exact (tree models) | Optimal for tree models |
| Gradient approximation (GradientSHAP) | $O(N \cdot B)$ | Depends on background samples | Deep learning |

where $K$ is the sample size, $T_{eval}$ is the model evaluation time, and $B$ is the number of background samples.

### Limitations and Considerations of SHAP

1. **Computational cost**: For high-dimensional data and complex models, the computational cost can be high
2. **Background selection sensitivity**: Different background data will lead to different explanations
3. **Feature correlation**: Attribution for strongly correlated features may be unstable
4. **Caution with causal explanations**: SHAP values are associative and cannot be directly interpreted as causal effects

Practical suggestions:
- Use representative background data
- For correlated features, consider grouping or using PartitionExplainer
- Validate the reasonableness of explanations with domain knowledge
- Use multiple visualizations for cross-validation

## Conclusion

SHAP, as a model explanation framework based on Shapley values, provides a theoretically rigorous and practically effective interpretability solution. Its main contributions include:

1. **Unified framework**: Unifying various explanation methods under an additive feature attribution framework
2. **Theoretical guarantees**: Ensuring consistency and fairness of explanations through the uniqueness theorem
3. **Local and global**: Supporting explanations for both individual predictions and global patterns
4. **Practical tools**: Providing rich visualization and analysis tools

SHAP is not just a tool for understanding models, but also a bridge for improving models, discovering insights, and building trust. As the importance of explainable AI becomes increasingly prominent, SHAP will play a key role in ensuring the transparency, fairness, and reliability of AI systems.
