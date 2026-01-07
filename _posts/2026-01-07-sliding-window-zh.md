---
title: 滑动窗口
author: lukeecust
date: 2026-01-07 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: zh
math: true
translation_id: sliding-window
permalink: /zh/sliding-window/
render_with_liquid: false
---

## 滑动窗口

### 209 长度最小的子数组：[leetcode.cn/problems/minimum-size-subarray-sum/](https://leetcode.cn/problems/minimum-size-subarray-sum)

给定一个含有 `n` 个正整数的数组和一个正整数 `target` 。

找出该数组中满足其总和大于等于 `target` 的长度最小的 **子数组** <code>[nums<sub>l</sub>, nums<sub>l+1</sub>, ..., nums<sub>r-1</sub>, nums<sub>r</sub>]</code>，并返回其长度。如果不存在符合条件的子数组，返回 `0` 。

**思路与解答（可变长窗口，最小化长度）**

关键条件：`nums` 全为正数，因此当我们右扩时窗口和单调不减；当我们左缩时窗口和单调不增。这使得双指针可以线性推进。

不变量设计：

* 维护窗口 `[l, r]` 的区间和 `s`。
* 当 `s >= target` 时，尝试不断右移 `l` 来缩短窗口，同时更新最优解。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        ans = inf
        s, l = 0, 0
        for r, x in enumerate(nums):
            s += x
            while s >= target:
                ans = min(ans, r - l + 1)
                s -= nums[l]
                l += 1
        return ans if ans <= len(nums) else 0
```

**复杂度分析**

* 时间复杂度：`O(n)`，`l`、`r` 都最多移动 `n` 次。
* 空间复杂度：`O(1)`。

### 713 乘积小于 `K` 的子数组：[leetcode.cn/problems/subarray-product-less-than-k](https://leetcode.cn/problems/subarray-product-less-than-k)

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回子数组内所有元素的乘积严格小于 `k` 的连续子数组的数目。

**思路与解答（可变长窗口，计数型）**

关键条件：`nums[i]` 为正整数时，窗口乘积在右扩时单调不减；左缩会单调不增，因此可用双指针维护“窗口乘积 < k”。

计数技巧：

* 设当前窗口为 `[l, r]` 且满足 `product < k`。
* 那么**所有以 `r` 结尾**的合法子数组个数为：`r - l + 1`（起点可选 `l..r`）。

边界：当 `k <= 1` 时，正整数乘积不可能 `< k`，直接返回 0。


```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        ans, s, l = 0, 1, 0
        for r, x in enumerate(nums):
            s *= nums[r]
            while s >= k and l <= r:
                s //= nums[l]
                l += 1
            ans += r - l + 1
        return ans  
```

**复杂度分析**

* 时间复杂度：`O(n)`。
* 空间复杂度：`O(1)`。
  

### 3 无重复字符的最长字串：[leetcode.cn/problems/longest-substring-without-repeating-characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters)

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长** **子串** 的长度。

**思路与解答（可变长窗口，最大化长度）**

不变量设计：

* 维护窗口 `[l, r]` 内“所有字符出现次数 ≤ 1”。
* 右扩加入 `s[r]` 后若出现重复，则不断左缩直至恢复不变量。
* 每次窗口合法时更新最大长度 `ans = max(ans, r - l + 1)`。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        cnt = Counter()
        l = 0
        for r, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] >= 2:
                cnt[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans 
```
