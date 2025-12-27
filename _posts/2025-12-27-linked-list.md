---
title: Linked List
author: lukeecust
date: 2025-12-26 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: en
math: true
translation_id: linked-list
permalink: /linked-list/hash/
render_with_liquid: false
---

## Linked List

### 1. Add Two Numbers: [leetcode.cn/problems/add-two-numbers/](https://leetcode.cn/problems/add-two-numbers/)

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit.

Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Approach and Solution (Linked List + Carry)**

1.  Use a "dummy head node" `head`, and let `l3` point to the tail of the current result list to facilitate inserting new nodes uniformly.
2.  Use a variable `s` to act as the "carry" (in each round, first add the current digits to get the new `s`, then use `s % 10` for the current digit and `s // 10` to update the carry).
3.  The loop condition is `l1 or l2 or s`: continue if either linked list has unprocessed digits or if there is still a carry.
4.  Value retrieval in each round: treat empty lists as `0`; after constructing the new node, move the pointers forward.

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = l3 = ListNode(0)
        s = 0
        while l1 or l2 or s:
            s = (l1.val if l1 else 0) + (l2.val if l2 else 0) + s
            l3.next = ListNode(s % 10)
            s = s // 10
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            l3 = l3.next 
        return head.next
```

**Complexity Analysis**

*   Time Complexity: `O(max(m, n))`, iterating through both linked lists digit by digit once.
*   Space Complexity: `O(max(m, n))`, the length of the result linked list is at most 1 digit longer than the longer input list (due to carry).
