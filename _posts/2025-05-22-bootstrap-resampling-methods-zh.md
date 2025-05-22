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

![bootstrap](https://lukeecust.github.io/blog/assets/images/2025-05-22-bootstrap-resampling-methods/discrepancy.png)
_左边为伪随机数组成的二维点集，右边则是低差异序列（如 Sobol 序列）点集，对整个空间的覆盖更加完整和均匀。_