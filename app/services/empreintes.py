"""
Empreintes documentaires — MinHash et LSH.
===========================================

Detecter qu'un document a deja ete analyse est plus subtil qu'il n'y parait.
Une comparaison octet a octet echoue des qu'une virgule change ; une
comparaison par embeddings est couteuse et confond deux cours *du meme
sujet* avec deux cours *identiques*.

La reponse classique est la **similarite de Jaccard** sur les n-grammes de
mots : deux documents sont proches s'ils partagent une grande part de leurs
sequences de mots. Calculee exactement, elle exige de conserver tous les
n-grammes de tous les documents.

**MinHash** (Broder, 1997) l'estime avec une empreinte de taille fixe : on
tire N permutations aleatoires de l'espace des n-grammes et on retient, pour
chacune, le plus petit indice atteint. La probabilite que deux documents
partagent la meme valeur sur une permutation est exactement leur similarite
de Jaccard — la moyenne sur N permutations en donne donc une estimation non
biaisee, avec une erreur en 1/racine(N).

**LSH par bandes** evite ensuite de comparer chaque nouveau document a tous
les precedents : l'empreinte est decoupee en bandes, deux documents ne sont
compares que s'ils tombent dans le meme seau sur au moins une bande. Sur une
poignee de documents c'est superflu ; a l'echelle d'un etablissement, c'est
ce qui rend la detection tenable.
"""

import hashlib
import re

import numpy as np

# Nombre de permutations : compromis entre precision (erreur ~ 1/racine(N))
# et taille de l'empreinte stockee avec chaque analyse.
NB_PERMUTATIONS = 128
# Longueur des n-grammes de mots. Cinq mots consecutifs sont assez
# discriminants pour un texte pedagogique, sans etre sensibles a une
# reformulation locale.
TAILLE_NGRAMME = 5
# Bandes LSH : 32 bandes de 4 lignes detectent avec une forte probabilite
# les paires au-dela de ~0,6 de similarite.
NB_BANDES = 32

_PREMIER = (1 << 61) - 1
_MOTS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+")


def _permutations(graine: int = 42):
    """Coefficients (a, b) des permutations affines h(x) = (a·x + b) mod p."""
    rng = np.random.default_rng(graine)
    a = rng.integers(1, _PREMIER - 1, size=NB_PERMUTATIONS, dtype=np.int64)
    b = rng.integers(0, _PREMIER - 1, size=NB_PERMUTATIONS, dtype=np.int64)
    return a, b


_A, _B = _permutations()


def _ngrammes(texte: str) -> set[int]:
    """N-grammes de mots, hachés en entiers 64 bits."""
    mots = _MOTS.findall((texte or "").lower())
    if len(mots) < TAILLE_NGRAMME:
        # Document trop court : on se rabat sur les mots isolés plutôt que de
        # renvoyer un ensemble vide, qui rendrait toute comparaison impossible.
        fragments = mots
    else:
        fragments = [
            " ".join(mots[i : i + TAILLE_NGRAMME])
            for i in range(len(mots) - TAILLE_NGRAMME + 1)
        ]
    return {
        int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big")
        % _PREMIER
        for f in fragments
    }


def empreinte(texte: str) -> list[int]:
    """
    Empreinte MinHash d'un document.

    Retourne une liste de `NB_PERMUTATIONS` entiers, directement
    serialisable et stockable avec l'analyse.
    """
    valeurs = _ngrammes(texte)
    if not valeurs:
        return [0] * NB_PERMUTATIONS

    x = np.fromiter(valeurs, dtype=np.int64, count=len(valeurs))
    # (a·x + b) mod p, vectorise sur toutes les permutations a la fois.
    hachages = (np.outer(_A, x) + _B[:, None]) % _PREMIER
    return hachages.min(axis=1).astype(np.int64).tolist()


def similarite(empreinte_a, empreinte_b) -> float:
    """
    Similarite de Jaccard estimee : part des permutations sur lesquelles les
    deux empreintes coincident.
    """
    a = np.asarray(empreinte_a, dtype=np.int64)
    b = np.asarray(empreinte_b, dtype=np.int64)
    if a.size != b.size or a.size == 0:
        return 0.0
    return float(np.mean(a == b))


def seaux_lsh(empreinte_donnee) -> list[str]:
    """
    Seaux LSH d'une empreinte : une signature courte par bande.

    Deux documents partageant au moins un seau sont candidats a la
    comparaison ; les autres sont ecartes sans calcul.
    """
    valeurs = np.asarray(empreinte_donnee, dtype=np.int64)
    if valeurs.size != NB_PERMUTATIONS:
        return []
    lignes = NB_PERMUTATIONS // NB_BANDES
    seaux = []
    for bande in range(NB_BANDES):
        tranche = valeurs[bande * lignes : (bande + 1) * lignes]
        signature = hashlib.blake2b(tranche.tobytes(), digest_size=8).hexdigest()
        seaux.append(f"{bande}:{signature}")
    return seaux


def chercher_proches(empreinte_donnee, candidats: list[dict], seuil: float = 0.55,
                     limite: int = 5) -> list[dict]:
    """
    Retrouve les documents proches parmi `candidats`.

    Chaque candidat doit porter les cles `empreinte` et, idealement,
    `seaux`. Le filtrage LSH est applique quand les seaux sont disponibles,
    sinon toutes les empreintes sont comparees — le resultat est identique,
    seul le cout differe.
    """
    seaux_reference = set(seaux_lsh(empreinte_donnee))
    resultats = []

    for candidat in candidats:
        signature = candidat.get("empreinte")
        if not signature:
            continue
        seaux_candidat = set(candidat.get("seaux") or [])
        if seaux_reference and seaux_candidat and not (seaux_reference & seaux_candidat):
            continue  # aucune bande commune : ecarte sans comparaison
        score = similarite(empreinte_donnee, signature)
        if score >= seuil:
            resultats.append({**candidat, "similarite": round(score, 3)})

    resultats.sort(key=lambda r: r["similarite"], reverse=True)
    return resultats[:limite]


def infos() -> dict:
    return {
        "algorithme": "MinHash (Broder, 1997) + LSH par bandes",
        "nb_permutations": NB_PERMUTATIONS,
        "taille_ngramme": TAILLE_NGRAMME,
        "nb_bandes": NB_BANDES,
        "erreur_estimation": round(1 / np.sqrt(NB_PERMUTATIONS), 3),
    }
