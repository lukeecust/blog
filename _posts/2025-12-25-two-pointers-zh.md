---
title: 双指针
author: lukeecust
date: 2025-12-25 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: zh
math: true
translation_id: journals-links
permalink: /zh/posts/two-pointers/
render_with_liquid: false
---

## 双指针

### 1 两数之和（Two Sum）：[leetcode.cn/problems/two-sum/](https://leetcode.cn/problems/two-sum/)

给定整数数组 `nums` 和目标值 `target`，在数组中找出两个数使得它们的和等于 `target`，并返回这两个数的下标。

* 你可以假设每种输入只会对应一个答案；
* 同一个元素不能使用两次；
* 下标返回顺序任意。

**思路与解答（双指针 + 排序）**

1. 双指针适用于有序数组；对于无序 `nums` 先排序。
2. 为了返回原下标：先得到“按值排序的索引序列” `idx_sorted`，再构造对应的 `nums_sorted`。
3. 在 `nums_sorted` 上用双指针：和小则 `l += 1`，和大则 `r -= 1`，相等时返回 `[idx_sorted[l], idx_sorted[r]]`。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # idx_sorted: 排序后元素在原数组中的下标序列
        idx_sorted = sorted(range(len(nums)), key=nums.__getitem__)

        # nums_sorted: 排序后的数值序列（与 idx_sorted 一一对应）
        nums_sorted = [nums[i] for i in idx_sorted]

        l, r = 0, len(nums_sorted) - 1
        while l < r:
            s = nums_sorted[l] + nums_sorted[r]
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                return [idx_sorted[l], idx_sorted[r]]

        return [-1, -1]
```

**复杂度分析**

* 时间复杂度：排序为$(O(n\log n))$，双指针扫描为$(O(n))$，总计$(O(n\log n))$。
* 空间复杂度：额外保存`idx_sorted`和`nums_sorted`，为$(O(n))$。

### 1 两数之和（Two Sum）：[leetcode.cn/problems/two-sum/](https://leetcode.cn/problems/two-sum/)

给定整数数组 `nums` 和目标值 `target`，在数组中找出两个数使得它们的和等于 `target`，并返回这两个数的下标。

* 你可以假设每种输入只会对应一个答案；
* 同一个元素不能使用两次；
* 下标返回顺序任意。

**思路与解答（双指针 + 排序）**

1. 双指针适用于有序数组；对于无序 `nums` 先排序。
2. 为了返回原下标：先得到“按值排序的索引序列” `idx_sorted`，再构造对应的 `nums_sorted`。
3. 在 `nums_sorted` 上用双指针：和小则 `l += 1`，和大则 `r -= 1`，相等时返回 `[idx_sorted[l], idx_sorted[r]]`。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # idx_sorted: 排序后元素在原数组中的下标序列
        idx_sorted = sorted(range(len(nums)), key=nums.__getitem__)

        # nums_sorted: 排序后的数值序列（与 idx_sorted 一一对应）
        nums_sorted = [nums[i] for i in idx_sorted]

        l, r = 0, len(nums_sorted) - 1
        while l < r:
            s = nums_sorted[l] + nums_sorted[r]
            if s < target:
                l += 1
            elif s > target:
                r -= 1
            else:
                return [idx_sorted[l], idx_sorted[r]]

        return [-1, -1]
```

**复杂度分析**

* 时间复杂度：排序为$(O(n\log n))$，双指针扫描为$(O(n))$，总计$(O(n\log n))$。
* 空间复杂度：额外保存`idx_sorted`和`nums_sorted`，为$(O(n))$。

### 3 三数之和：[https://leetcode.cn/problems/3sum/](https://leetcode.cn/problems/3sum/)

给定整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k`、`j != k`，且 `nums[i] + nums[j] + nums[k] == 0`。返回所有和为 `0` 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

**思路与解答（排序 + 双指针）**

1. 先排序，将三数之和转化为：枚举一个数 `nums[i]`，在区间 `(i, n-1]` 内做“两数之和 = -nums[i]`”的双指针搜索。
2. 去重：`i` 去重；命中答案后对 `j`、`k` 跳过连续重复值。
3. 剪枝：若当前最小三数和已大于 0，直接结束；若当前最大三数和仍小于 0，跳过该 `i`。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)

        for i in range(n - 2):
            # i 去重
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 剪枝：最小三数和 > 0，后面只会更大
            if nums[i] + nums[i + 1] + nums[i + 2] > 0:
                break
            # 剪枝：最大三数和 < 0，这个 i 不可能有解
            if nums[i] + nums[n - 2] + nums[n - 1] < 0:
                continue

            target = -nums[i]
            j, k = i + 1, n - 1
            while j < k:
                s = nums[j] + nums[k]
                if s < target:
                    j += 1
                elif s > target:
                    k -= 1
                else:
                    res.append([nums[i], nums[j], nums[k]])

                    # j, k 去重
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1

        return res
```

