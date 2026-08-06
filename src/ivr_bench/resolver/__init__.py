"""Resolution du praticien, strictement separee du routage."""

from ivr_bench.resolver.normalization import comparable, normalize, strip_titles
from ivr_bench.resolver.phonetics import phonetic_key, phonetic_keys, strip_accents
from ivr_bench.resolver.resolver import PractitionerResolver

__all__ = [
    "PractitionerResolver",
    "comparable",
    "normalize",
    "phonetic_key",
    "phonetic_keys",
    "strip_accents",
    "strip_titles",
]
