"""
Synthese extractive — TextRank et MMR.
=======================================

La synthese executive du rapport est aujourd'hui redigee par un modele de
langage. C'est efficace, mais cela cree une dependance a un service externe
pour une fonction que l'utilisateur percoit comme centrale : quand l'API est
indisponible, le rapport s'ouvre sur un gabarit.

Ce module produit une synthese **extractive** : au lieu de generer un texte
nouveau, il selectionne les phrases du document qui le representent le mieux.
Aucun modele de langage, aucun appel reseau, un resultat reproductible — la
meme entree donne toujours la meme synthese, ce qui compte pour un document
destine a un dossier d'accreditation.

TextRank
--------

L'algorithme (Mihalcea et Tarau, 2004) transpose PageRank au texte. On
construit un graphe dont les sommets sont les phrases et ou deux phrases sont
reliees proportionnellement a leur similarite. On y fait ensuite circuler une
« importance » : une phrase est importante si elle ressemble a beaucoup de
phrases elles-memes importantes. C'est une definition circulaire, resolue par
convergence — exactement le principe de PageRank sur le graphe du Web.

L'interet ici est que l'importance n'est pas une heuristique de position (la
premiere phrase, la derniere) mais une propriete emergente du contenu : la
phrase centrale d'un cours est celle qui fait echo au reste du cours.

MMR — Maximal Marginal Relevance
--------------------------------

Prendre les N phrases les mieux classees par TextRank produit une synthese
redondante : les phrases centrales se ressemblent, c'est precisement pourquoi
elles sont centrales. MMR (Carbonell et Goldstein, 1998) corrige ce biais en
selectionnant a chaque etape la phrase qui maximise

    lambda x pertinence  -  (1 - lambda) x similarite maximale aux phrases deja retenues

Le parametre lambda arbitre entre representativite et diversite. La synthese
obtenue couvre le document au lieu de repeter son centre.
"""

import re

import numpy as np

# Poids accorde a la pertinence face a la diversite dans MMR. 0,72 privilegie
# la representativite tout en ecartant les quasi-doublons.
LAMBDA_MMR = 0.72
# Bornes de longueur des phrases retenues : en deca on capte des fragments de
# titre, au-dela des enumerations entieres.
MOTS_MIN_PHRASE = 8
MOTS_MAX_PHRASE = 60
# Amortissement de PageRank, valeur usuelle.
AMORTISSEMENT = 0.85
ITERATIONS_MAX = 60
TOLERANCE = 1e-6

_FIN_PHRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂÎÔÛÇ0-9])")
_ESPACES = re.compile(r"\s+")


def decouper_phrases(texte: str) -> list[str]:
    """Segmente un texte en phrases exploitables."""
    texte = _ESPACES.sub(" ", (texte or "").strip())
    if not texte:
        return []
    phrases = []
    for brut in _FIN_PHRASE.split(texte):
        phrase = brut.strip(" -•\t")
        nb_mots = len(phrase.split())
        if MOTS_MIN_PHRASE <= nb_mots <= MOTS_MAX_PHRASE:
            phrases.append(phrase)
    return phrases


def _similarites(phrases: list[str]) -> tuple[np.ndarray, str]:
    """
    Matrice de similarite entre phrases.

    On reutilise l'encodeur semantique du pipeline quand il est disponible :
    deux phrases qui disent la meme chose avec d'autres mots doivent etre
    reliees. A defaut, un recouvrement lexical normalise joue le meme role de
    facon plus grossiere mais sans aucune dependance.
    """
    from app.services import entrainement

    vecteurs, source = entrainement.encoder_textes(phrases)
    if vecteurs is not None:
        normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
        normes[normes == 0] = 1.0
        unitaires = vecteurs / normes
        return np.clip(unitaires @ unitaires.T, 0.0, 1.0), source

    # Repli : similarite de recouvrement ponderee par la longueur, telle que
    # definie dans l'article original de TextRank.
    ensembles = [set(re.findall(r"[a-zà-öø-ÿ]{3,}", p.lower())) for p in phrases]
    n = len(phrases)
    matrice = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            commun = len(ensembles[i] & ensembles[j])
            if commun:
                denominateur = np.log(len(ensembles[i]) + 1) + np.log(len(ensembles[j]) + 1)
                valeur = commun / denominateur if denominateur else 0.0
                matrice[i, j] = matrice[j, i] = valeur
    maximum = matrice.max() or 1.0
    return matrice / maximum, "recouvrement lexical (repli)"


def textrank(similarites: np.ndarray) -> np.ndarray:
    """
    Score d'importance de chaque phrase, par PageRank sur le graphe.

    La diagonale est annulee — une phrase ne se recommande pas elle-meme —
    puis chaque ligne est normalisee pour former une matrice stochastique.
    """
    n = similarites.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    matrice = similarites.copy()
    np.fill_diagonal(matrice, 0.0)
    sommes = matrice.sum(axis=1, keepdims=True)
    sommes[sommes == 0] = 1.0
    transition = matrice / sommes

    scores = np.full(n, 1.0 / n)
    for _ in range(ITERATIONS_MAX):
        suivant = (1 - AMORTISSEMENT) / n + AMORTISSEMENT * (transition.T @ scores)
        if np.abs(suivant - scores).sum() < TOLERANCE:
            scores = suivant
            break
        scores = suivant
    return scores


def selectionner_mmr(similarites: np.ndarray, pertinence: np.ndarray,
                     nb: int, lambda_: float = LAMBDA_MMR) -> list[int]:
    """Selection gloutonne par Maximal Marginal Relevance."""
    n = len(pertinence)
    nb = min(nb, n)
    if nb <= 0:
        return []

    # Normalisation : pertinence et similarite doivent etre comparables pour
    # que lambda ait le sens d'un arbitrage.
    etendue = pertinence.max() - pertinence.min()
    normalisee = (
        (pertinence - pertinence.min()) / etendue if etendue > 1e-12
        else np.ones_like(pertinence)
    )

    retenus = [int(np.argmax(normalisee))]
    while len(retenus) < nb:
        meilleur, score_max = None, -np.inf
        for i in range(n):
            if i in retenus:
                continue
            redondance = max(similarites[i][j] for j in retenus)
            score = lambda_ * normalisee[i] - (1 - lambda_) * redondance
            if score > score_max:
                meilleur, score_max = i, score
        if meilleur is None:
            break
        retenus.append(meilleur)
    return retenus


def resumer(texte: str, nb_phrases: int = 5) -> dict:
    """
    Synthese extractive d'un texte.

    Les phrases retenues sont restituees **dans leur ordre d'apparition** :
    une synthese qui suit le fil du document se lit mieux qu'une liste
    classee par score.
    """
    phrases = decouper_phrases(texte)
    if len(phrases) < 2:
        return {
            "disponible": False,
            "motif": "texte trop court pour une synthèse extractive",
            "phrases": phrases, "resume": " ".join(phrases),
            "nb_phrases_source": len(phrases),
        }

    similarites, source = _similarites(phrases)
    scores = textrank(similarites)
    retenus = sorted(selectionner_mmr(similarites, scores, nb_phrases))

    selection = [
        {
            "rang": position + 1,
            "phrase": phrases[i],
            "score": round(float(scores[i]), 4),
            "position_source": i,
        }
        for position, i in enumerate(retenus)
    ]
    return {
        "disponible": True,
        "resume": " ".join(s["phrase"] for s in selection),
        "phrases": selection,
        "nb_phrases_source": len(phrases),
        "nb_phrases_retenues": len(selection),
        "taux_compression": round(1 - len(selection) / len(phrases), 3),
        "algorithme": "TextRank (PageRank sur graphe de phrases) + MMR",
        "similarite": source,
        "lambda_mmr": LAMBDA_MMR,
    }


def resumer_analyse(analyse: dict, nb_phrases: int = 5) -> dict:
    """
    Synthese extractive du support de cours analyse.

    La source est le texte reellement extrait par l'Agent 1 : la synthese
    porte donc sur le document tel que le systeme l'a lu, pas sur une
    reformulation.
    """
    agent1 = analyse.get("agent1") or {}
    texte = agent1.get("texte_complet") or agent1.get("texte_brut_tronque") or ""

    if not texte:
        # Repli : reconstituer un texte a partir des unites de l'Agent 3,
        # qui survivent a la persistance la ou le texte complet est ecarte.
        unites = ((analyse.get("agent3") or {}).get("unites") or [])
        texte = " ".join(u.get("texte", "") for u in unites)

    resultat = resumer(texte, nb_phrases)
    resultat["source_texte"] = (
        "texte extrait par l'Agent 1" if agent1.get("texte_complet")
        else "unités de contenu de l'Agent 3"
    )
    return resultat
