---
title: Sobol的准随机序列生成器
author: lukeecust
date: 2025-05-21 02:09:00 +0800
categories: [Deep Learning, 待补充]
tags: [准随机序列]
lang: zh
math: true
translation_id: sobol-sequence-generator
permalink: /zh/posts/sobol-sequence-generator/
render_with_liquid: false
---

Sobol序列是低差异的准随机数。它旨在更均匀地覆盖单位超立方体，并且在高维问题中特别有效。其他值得注意的序列包括 Faure 序列和 Niederreiter 序列，每种序列都根据应用的具体要求提供独特的优势。

所有基于现代 CPU 的随机数生成算法都是伪随机的（quasi-random）。它们受限于一个周期。当超过周期后就会重复出现，而不再是相互无关的随机数。这个周期的最终限定是由电脑的位数来决定的，因此，没有一个内建的随机数是“真正”随机的。
　　Sobol 采样使用了不同的方式来采样。比起随机数，Sobol 序列着重于在概率空间中产生均匀的分布。但这并不是单纯的使用网格来填满，而是使用一个本质上随机，但是巧妙的方法去“填满”概率空间，即之后产生的随机数会分布到之前没有采样到的区域。

高效的生成在高维空间分布均匀的随机数是在计算机程序中非常常见的组成部分。对于一切需要采样的算法来说，分布均匀的随机数就意味着更加优秀的样本分布。光线传递的模拟（渲染）基于蒙特卡洛积分（Monte Carlo Integration），这个过程中采样无处不在，所以好的样本分布直接影响积分过程的收敛速度。与常见的伪随机数对比，低差异序列（Low Discrepancy Sequence）非常广泛的被用在图形，甚至于金融领域。

什么是Discrepancy

![Desktop View](https://lukeecust.github.io/blog/assets/images/2025-05-21-sobol-sequence-generator/discrepancy.png){:.left }
_伪随机和低差异序列_

左边为伪随机数组成的二维点集，右边则是由低差异序列点集的对整个空间的覆盖更加完整。

$$\begin{equation}
D_N(P)=\sup _{B \in J}\left|\frac{A(B)}{N}-\lambda_s(B)\right|
\end{equation}$$

对于一个在 $[0,1]^n$ 空间中的点集，任意选取一个空间中的“区间” $B$ ，此区域内点的数量 $A$ 和点集个数的总量 $N$ 的比值和此区域的体积 $\lambda_s$ 的差的绝对值的最大值，就是这个点集的Discrepancy。分布越均匀的点集，任意区域内的点集数量占点总数量的比例也会越接近于这个区域的体积。