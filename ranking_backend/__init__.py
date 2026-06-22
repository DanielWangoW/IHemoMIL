# The heavy lifting lives in the compiled extension `_ranking_impl` (built from
# _ranking_impl.pyx). 
from ihemomil_ranking._ranking_impl import (  # type: ignore
    RankAveragePooling,
    RankWeightPooling,
    MILRankAdditivePooling,
    MILRankConjunctivePooling,
)

__all__ = ["get_pooling"]

_POOLING_BY_KEY = {
    "rap": RankAveragePooling,
    "ratte": RankWeightPooling,
    "raddi": MILRankAdditivePooling,
    "rconj": MILRankConjunctivePooling,
}


def get_pooling(key):
    """Return the compiled pooling class registered under ``key`` (e.g. 'rconj'),
    or ``None`` if the key is unknown."""
    return _POOLING_BY_KEY.get(key)
