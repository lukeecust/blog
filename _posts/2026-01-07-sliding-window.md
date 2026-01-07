---
title: Sliding Window
author: lukeecust
date: 2026-01-07 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: en
math: true
translation_id: sliding-window
permalink: sliding-window/
render_with_liquid: false
---

## Sliding Window

### 209. Minimum Size Subarray Sum: [leetcode.cn/problems/minimum-size-subarray-sum/](https://leetcode.cn/problems/minimum-size-subarray-sum)

Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a **subarray** <code>[nums<sub>l</sub>, nums<sub>l+1</sub>, ..., nums<sub>r-1</sub>, nums<sub>r</sub>]</code> of which the sum is greater than or equal to `target`. If there is no such subarray, return `0` instead.

**Intuition & Solution (Variable Length Window, Minimize Length)**

Key Condition: `nums` consists of all positive numbers. Thus, when we expand the window to the right, the window sum is monotonically non-decreasing; when we shrink from the left, the window sum is monotonically non-increasing. This allows the two pointers to move linearly.

Invariant Design:

* Maintain the interval sum `s` of the window `[l, r]`.
* When `s >= target`, attempt to continuously move `l` to the right to shrink the window while updating the optimal solution.

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

**Complexity Analysis**

* Time Complexity: `O(n)`, both `l` and `r` move at most `n` times.
* Space Complexity: `O(1)`.

### 713. Subarray Product Less Than K: [leetcode.cn/problems/subarray-product-less-than-k](https://leetcode.cn/problems/subarray-product-less-than-k)

Given an array of integers `nums` and an integer `k`, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than `k`.

**Intuition & Solution (Variable Length Window, Counting)**

Key Condition: Since `nums[i]` are positive integers, the window product is monotonically non-decreasing when expanding to the right; it is monotonically non-increasing when shrinking from the left, so two pointers can be used to maintain "window product < k".

Counting Technique:

* Let the current window be `[l, r]` satisfying `product < k`.
* Then the number of valid subarrays **ending at `r`** is: `r - l + 1` (start points can be chosen from `l..r`).

Boundary: When `k <= 1`, the product of positive integers cannot be `< k`, so directly return 0.

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

**Complexity Analysis**

* Time Complexity: `O(n)`.
* Space Complexity: `O(1)`.
  

### 3. Longest Substring Without Repeating Characters: [leetcode.cn/problems/longest-substring-without-repeating-characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters)

Given a string `s`, find the length of the **longest substring** without repeating characters.

**Intuition & Solution (Variable Length Window, Maximize Length)**

Invariant Design:

* Maintain that "all characters appear at most once" within the window `[l, r]`.
* After expanding to the right by adding `s[r]`, if a duplicate appears, continuously shrink from the left until the invariant is restored.
* Each time the window is valid, update the maximum length `ans = max(ans, r - l + 1)`.

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
