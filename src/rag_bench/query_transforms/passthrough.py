from __future__ import annotations

from .base import QueryTransformer, register


@register("baseline")
class PassthroughTransformer(QueryTransformer):
    """No transformation — returns the original query as-is.

    This is the baseline. Equivalent to current pipeline behavior.
    """

    def transform(self, question: str) -> list[str]:
        return [question]