"""Arguments enumeres : une classification, pas une extraction.

`topic`, `reason` et `reason_category` ne figurent nulle part dans la phrase.
« vous prenez la carte vitale ? » attend `accepted_insurance`, un mot que
l'enonce ne contient pas. Aucune extraction de segment ne peut donc les
trouver — ni les regles, ni DIET, ni un modele d'appel d'outils qui recopie du
texte. L'extracteur partage repond une constante, et sur `topic`, quatorze
valeurs possibles, cette constante n'est jamais la bonne.

Ce composant repond a la seule question qui se pose reellement : parmi les
valeurs declarees au catalogue, laquelle cette phrase demande-t-elle ? Un
classifieur lexical par argument, entraine sur `train` — le meme corpus que
toutes les architectures apprenantes — suffit a la poser correctement.

Il est isole ici parce qu'il ne depend d'aucune architecture : A16 l'utilise
apres DIET, A17 sans DIET. C'est ce qui permet de mesurer separement ce que
chacun apporte.
"""

from __future__ import annotations

from typing import Any

from ivr_bench.domain.models import ToolDefinition
from ivr_bench.generators.corpus import load_split

# Arguments dont la valeur appartient a une enumeration du catalogue.
ENUMERATED = frozenset({"topic", "reason", "reason_category"})


class EnumerationLearner:
    """Un classifieur par couple (fonction, argument enumere)."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._models)

    def fit(self, train_split: str = "train", seed: int = 42) -> None:
        """Entrainement sur `train` seul, comme tout composant appris du banc."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        grouped: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for case in load_split(train_split):
            function = case.expected.tool_name
            for name, value in case.expected.arguments.items():
                if name not in ENUMERATED or not isinstance(value, str) or not value:
                    continue
                grouped.setdefault((function, name), []).append((case.utterance, value))

        for key, examples in grouped.items():
            texts = [text for text, _ in examples]
            labels = [label for _, label in examples]
            if len(set(labels)) < 2:
                # Une seule valeur observee : un modele n'apprendrait rien de
                # plus qu'une constante, et il faut le dire ainsi.
                self._models[key] = labels[0]
                continue
            model = make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            model.fit(texts, labels)
            self._models[key] = model

    def value(self, definition: ToolDefinition, name: str, utterance: str) -> Any:
        """Valeur predite, ou la valeur par defaut si rien n'a ete appris."""
        parameter = definition.parameter(name)
        if parameter is None or not parameter.enum:
            return None
        model = self._models.get((definition.name, name))
        if model is None:
            return parameter.enum[-1]
        if isinstance(model, str):
            return model
        return str(model.predict([utterance])[0])
