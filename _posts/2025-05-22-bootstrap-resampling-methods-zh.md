---
title: Bootstrap 重采样方法
author: lukeecust
date: 2025-05-22 14:09:00 +0800
categories: [Data Science, Sampling]
tags: [resampling]
lang: zh
math: true
translation_id: bootstrap-resampling-methods
permalink: /zh/posts/bootstrap-resampling-methods/
render_with_liquid: false
---

bootstrap 的基本思想是在样本数据上对总体参数 $\theta$（如总体平均值）的**估计值**（如样本平均值）进行推断。它是一种**重采样**方法，从具有相同样本量 $n$ 的现有样本数据中独立地进行替换采样，并在这些重采样数据中进行推断。


一般来说，bootstrap涉及以下步骤：
![bootstrap](https://lukeecust.github.io/blog/assets/images/2025-05-22-bootstrap-resampling-methods/bootstrap.png){: .w-50 .left }
_bootstrap的步骤_

1．从总体中抽取的样本，样本量为 $n$ 。
2．从样本量为 $n$ 的原始样本数据中抽取一个样本，重复 $B$ 次，每个重复抽取的样本称为一个 Bootstrap 样本，总共有 $B$ 个 Bootstrap 样本。
3．计算每个 Bootstrap 样本的统计量，共有 $B$ 个 ${ }^* \theta$ 的估计值。
4．用这 $B$ 个 Bootstrap 统计量构建一个抽样分布，并用它来做进一步的统计推断，例如：
- 估计 $\theta$ 统计量的标准误差
- 获取 $\theta$ 的置信区间

我们可以看到，我们通过从现有样本中重新取样来生成新的数据点，并根据这些新的数据点进行推断。