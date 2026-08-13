"""
Re-ranking par cross-encodeur.
===============================

Le bi-encodeur de l'Agent 4 encode le cours et les notions **separement**,
puis compare les vecteurs obtenus. Cette independance est ce qui rend la
recherche rapide (les notions sont vectorisees une fois pour toutes et
indexees dans FAISS), mais c'est aussi sa faiblesse : le modele mesure une
proximite **thematique** et non une **implication pedagogique**. Un cours qui
mentionne les fractions obtient un score eleve face a la notion « conversion
fraction / pourcentage » alors qu'il ne l'enseigne pas.

Un cross-encodeur traite la paire (texte du cours, notion) **conjointement**,
en une seule passe d'attention. Il ne peut pas etre pre-indexe — donc il est
lent — mais il est nettement plus precis. D'ou l'architecture classique en
deux etages : le bi-encodeur rappelle des candidats, le cross-encodeur
tranche entre eux.

Ce module est partage :
- l'Agent 6 l'utilise pour decider de la couverture d'une notion ;
- l'Agent 8 l'utilise pour verifier qu'un exercice genere porte bien sur la
  notion visee.
"""

import math

import numpy as np

from app.config import Config
from app.services import model_registry

# Calibration par defaut, utilisee tant que le classifieur entraine
# (`couverture_clf`) n'a pas ete depose dans `app/models/`.
#
# Le modele `mmarco-mMiniLMv2` renvoie des logits non bornes, et leur echelle
# depend fortement de la longueur des textes compares : de courtes phrases
# temoins produisent des logits tres negatifs, la ou de vraies unites de cours
# se situent autour de zero. La calibration ci-dessous a donc ete etablie sur
# la distribution reellement observee lors de l'analyse d'un support de cours
# (55 notions de reference, 5 pays) :
#
#     percentile  0 : -2,06      percentile 75 : +0,34
#     percentile 25 : -0,63      percentile 90 : +1,72
#     percentile 50 : -0,24      percentile 100 : +3,84
#
# Les notions effectivement enseignees se detachaient au-dessus de +0,3, les
# notions absentes du support restant sous -0,2. Le point milieu est donc
# fixe a +0,35, avec une echelle de 0,9 qui couvre l'essentiel de la plage.
#
# Ce sont des constantes d'ingenierie, explicitees et reproductibles, mais ce
# ne sont PAS des parametres appris : elles seront remplacees par le
# classifieur calibre issu du notebook `01_couverture.ipynb`.
POINT_MILIEU = float(0.35)
ECHELLE = float(0.90)


def est_actif() -> bool:
    """Le cross-encodeur est-il chargeable et active par configuration ?"""
    if not Config.USE_CROSS_ENCODER:
        return False
    return model_registry.est_disponible("cross_encoder")


def scorer_paires(paires: list[tuple[str, str]], taille_lot: int = 32) -> np.ndarray | None:
    """
    Score une liste de paires (texte_a, texte_b).

    Retourne un tableau de logits, ou `None` si le cross-encodeur n'est pas
    disponible — l'appelant bascule alors sur la similarite cosinus seule.
    """
    if not paires:
        return np.zeros(0, dtype="float32")

    modele = model_registry.charger("cross_encoder")
    if modele is None:
        return None

    try:
        scores = modele.predict(paires, batch_size=taille_lot, show_progress_bar=False)
        return np.asarray(scores, dtype="float32").reshape(-1)
    except Exception:
        # Une defaillance du re-ranking ne doit jamais interrompre l'analyse.
        return None


def logit_vers_probabilite(logits) -> np.ndarray:
    """Convertit les logits bruts du cross-encodeur en pseudo-probabilites."""
    valeurs = np.asarray(logits, dtype="float64").reshape(-1)
    return 1.0 / (1.0 + np.exp(-(valeurs - POINT_MILIEU) / ECHELLE))


def probabilite_unique(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-(float(logit) - POINT_MILIEU) / ECHELLE))


def infos() -> dict:
    """Description du moteur de re-ranking, restituee dans le rapport."""
    actif = est_actif()
    return {
        "actif": actif,
        "modele": Config.CROSS_ENCODER_MODEL if actif else None,
        "top_k": Config.CROSS_ENCODER_TOP_K,
        "calibration": {"point_milieu": POINT_MILIEU, "echelle": ECHELLE},
        "role": (
            "Re-score conjointement chaque paire (unité de cours, notion) issue de la "
            "recherche vectorielle, afin d'écarter les correspondances thématiques "
            "qui ne correspondent pas à un réel enseignement de la notion."
        ),
    }
