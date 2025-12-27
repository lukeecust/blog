---
title: 链表
author: lukeecust
date: 2025-12-26 00:09:00 +0800
categories: [LeetCode, Hot100]
lang: zh
math: true
translation_id: linked-list
permalink: /zh/linked-list/hash/
render_with_liquid: false
---

## 链表

### 1 两数相加（Add Two Numbers）：[leetcode.cn/problems/add-two-numbers/](https://leetcode.cn/problems/add-two-numbers/)

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 `0` 之外，这两个数都不会以 `0` 开头。

**思路与解答（链表 + 进位）**

1. 用一个“虚拟头结点” `head`，`l3` 指向当前结果链表的尾部，便于统一插入新节点。
2. 用变量 `s` 充当“进位 carry”（每轮先把当前位相加得到新 `s`，再用 `s % 10` 取当前位、`s // 10` 更新进位）。
3. 循环条件用 `l1 or l2 or s`：两条链表有没处理完的位，或者还有进位，就继续。
4. 每轮取值：链表为空就当作 ```0```；构造新节点后指针后移。

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

**复杂度分析**

* 时间复杂度：`O(max(m, n))`，两条链表逐位相加一次。
* 空间复杂度：`O(max(m, n))`，结果链表长度最多比更长的那条多 1 位（进位）。

