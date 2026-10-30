"""Minimal union-find (disjoint sets) over integer indices 0..n-1."""

from __future__ import annotations


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def labels(self) -> list[int]:
        """Component label per index, numbered 0.. in order of first appearance."""
        remap: dict[int, int] = {}
        out = []
        for i in range(len(self.parent)):
            root = self.find(i)
            out.append(remap.setdefault(root, len(remap)))
        return out
