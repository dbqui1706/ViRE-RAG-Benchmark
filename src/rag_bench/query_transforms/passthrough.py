from __future__ import annotations
from .base import QueryTransformer, register

@register("passthrough")
def _factory(**kwargs) -> PassthroughTransformer:
    return PassthroughTransformer()

class PassthroughTransformer(QueryTransformer):
    def batch_transform(self, queries: list[str]) -> list[list[str]]:
        return [[q] for q in queries]
