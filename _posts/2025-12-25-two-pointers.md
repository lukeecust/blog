---
title: Two Pointers
author: lukeecust
date: 2025-12-25 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: en
math: true
translation_id: journals-links
permalink: /posts/two-pointers/
render_with_liquid: false
---

## Two Pointers

### 1. Two Sum: [leetcode.cn/problems/two-sum/](https://leetcode.cn/problems/two-sum/)

Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.

*   You may assume that each input would have exactly one solution.
*   You may not use the same element twice.
*   You can return the answer in any order.

**Approach and Solution (Two Pointers + Sorting)**

1.  Two pointers work best on sorted arrays; for an unsorted `nums`, sort it first.
2.  To return the original indices: first obtain the "index sequence sorted by value" `idx_sorted`, then construct the corresponding `nums_sorted`.
3.  Use two pointers on `nums_sorted`: if the sum is smaller, `l += 1`; if larger, `r -= 1`; if equal, return `[idx_sorted[l], idx_sorted[r]]`.

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # idx_sorted: sequence of indices from the original array after sorting
        idx_sorted = sorted(range(len(nums)), key=nums.__getitem__)

        # nums_sorted: sorted sequence of values (corresponds one-to-one with idx_sorted)
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

**Complexity Analysis**

*   Time Complexity: Sorting is $(O(n\log n))$, two-pointer scan is $(O(n))$, total is $(O(n\log n))$.
*   Space Complexity: Storing `idx_sorted` and `nums_sorted` takes $(O(n))$.

### 3. 3Sum: [https://leetcode.cn/problems/3sum/](https://leetcode.cn/problems/3sum/)

Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` such that `i != j`, `i != k`, and `j != k`, and `nums[i] + nums[j] + nums[k] == 0`.

Notice that the solution set must not contain duplicate triplets.

**Approach and Solution (Sorting + Two Pointers)**

1.  Sort first, transforming the 3Sum problem into: enumerate a number `nums[i]`, and perform a "Two Sum = -nums[i]" two-pointer search in the range `(i, n-1]`.
2.  Deduplication: deduplicate `i`; after finding a solution, skip consecutive duplicate values for `j` and `k`.
3.  Pruning: If the current minimum sum of three numbers is greater than 0, terminate immediately; if the current maximum sum of three numbers is still less than 0, skip this `i`.

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        n = len(nums)

        for i in range(n - 2):
            # Deduplicate i
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # Pruning: minimum 3-sum > 0, subsequent sums will only be larger
            if nums[i] + nums[i + 1] + nums[i + 2] > 0:
                break
            # Pruning: maximum 3-sum < 0, no solution possible for this i
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

                    # Deduplicate j, k
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1

        return res
```
