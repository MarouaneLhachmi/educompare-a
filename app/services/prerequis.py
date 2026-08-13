"""
Graphe de prerequis entre notions.
===================================

Un parcours d'amelioration n'a de sens que s'il respecte l'ordre naturel des
apprentissages : proposer « resoudre des equations du premier degre » a un
eleve qui ne maitrise pas encore les operations sur les decimaux est
pedagogiquement absurde, meme si l'ecart mesure y est plus important.

L'Agent 8 a donc besoin d'un graphe orientant les notions les unes par
rapport aux autres. Ce module le construit **sans aucun appel a un modele de
langage**, a partir de deux signaux objectifs :

1. **L'ordre du referentiel officiel.** Les programmes scolaires sont rediges
   dans l'ordre d'enseignement : au sein d'un meme referentiel, une notion
   citee avant une autre en constitue potentiellement un prerequis. C'est
   une convention forte et verifiable, pas une supposition.

2. **La proximite semantique**, mesuree sur les vecteurs deja produits par
   l'Agent 4. Deux notions eloignees thematiquement n'entretiennent aucune
   relation de prerequis, meme si l'une precede l'autre dans le programme :
   « Litteratie financiere » ne conditionne pas « Geometrie ».

Le graphe obtenu est **acyclique par construction** (les aretes vont toujours
d'un indice inferieur vers un indice superieur, au sein d'un meme pays), ce
qui garantit qu'un ordre topologique existe toujours.
"""

import numpy as np

# Proximite minimale pour qu'une notion anterieure soit retenue comme
# prerequis. En dessous, les deux notions relevent de domaines distincts.
SEUIL_PROXIMITE = 0.45
# Au-dela de deux prerequis, le graphe devient trop contraint et le
# planificateur ne peut plus rien ordonnancer.
MAX_PREREQUIS = 2
# Une notion trop eloignee dans le programme n'est plus un prerequis direct.
PORTEE_MAX = 5


def construire(notions: list[dict], vecteurs: np.ndarray | None) -> dict:
    """
    Construit le graphe de prerequis.

    `notions` doit etre la liste a plat produite par
    `referentiels.notions_a_plat()`, dans l'ordre du fichier de reference —
    c'est cet ordre qui porte l'information pedagogique.

    Retourne :
    {
        "prerequis": {cle_notion: [cle_prerequis, ...]},
        "successeurs": {cle_notion: [cle_successeur, ...]},
        "ordre_topologique": [cle_notion, ...],
        "methode": str, "nb_aretes": int
    }
    """
    cles = [f"{n['code']}::{n['notion']}" for n in notions]
    prerequis: dict[str, list[str]] = {cle: [] for cle in cles}

    proximite_disponible = (
        vecteurs is not None and len(vecteurs) == len(notions) and len(notions) > 1
    )
    if proximite_disponible:
        # Les vecteurs de l'Agent 4 sont deja normalises : le produit scalaire
        # est directement la similarite cosinus.
        similarites = np.asarray(vecteurs) @ np.asarray(vecteurs).T
    else:
        similarites = None

    # Indices des notions, regroupees par referentiel (pays).
    par_pays: dict[str, list[int]] = {}
    for index, notion in enumerate(notions):
        par_pays.setdefault(notion["code"], []).append(index)

    for indices in par_pays.values():
        for rang, index in enumerate(indices):
            candidats = []
            # On ne remonte que de quelques positions : au-dela, la relation
            # de prerequis direct n'a plus de sens.
            debut = max(0, rang - PORTEE_MAX)
            for rang_amont in range(debut, rang):
                index_amont = indices[rang_amont]
                if similarites is not None:
                    proximite = float(similarites[index][index_amont])
                    if proximite < SEUIL_PROXIMITE:
                        continue
                else:
                    # Sans vecteurs, on se limite a la notion immediatement
                    # precedente : c'est le minimum defendable.
                    if rang_amont != rang - 1:
                        continue
                    proximite = 1.0
                candidats.append((proximite, cles[index_amont]))

            candidats.sort(reverse=True)
            prerequis[cles[index]] = [cle for _, cle in candidats[:MAX_PREREQUIS]]

    successeurs: dict[str, list[str]] = {cle: [] for cle in cles}
    nb_aretes = 0
    for cle, amonts in prerequis.items():
        for amont in amonts:
            successeurs[amont].append(cle)
            nb_aretes += 1

    return {
        "prerequis": prerequis,
        "successeurs": successeurs,
        "ordre_topologique": _ordre_topologique(cles, prerequis),
        "methode": (
            "ordre du référentiel officiel + proximité sémantique (cosinus ≥ "
            f"{SEUIL_PROXIMITE})"
            if proximite_disponible
            else "ordre du référentiel officiel (vecteurs indisponibles)"
        ),
        "nb_aretes": nb_aretes,
        "seuil_proximite": SEUIL_PROXIMITE,
    }


def _ordre_topologique(cles: list[str], prerequis: dict[str, list[str]]) -> list[str]:
    """Tri topologique (Kahn). Le graphe etant acyclique par construction,
    l'ordre existe toujours ; le repli ne sert que de garde-fou."""
    restants = {cle: set(prerequis.get(cle, [])) for cle in cles}
    ordre: list[str] = []
    while restants:
        libres = sorted(cle for cle, amonts in restants.items() if not amonts)
        if not libres:  # cycle inattendu : on vide dans l'ordre d'origine
            ordre.extend(cle for cle in cles if cle not in ordre)
            break
        for cle in libres:
            ordre.append(cle)
            del restants[cle]
        for amonts in restants.values():
            amonts.difference_update(libres)
    return ordre


def maitrise_prerequis(cle: str, graphe: dict, maitrises: dict[str, float]) -> float:
    """
    Niveau de maitrise moyen des prerequis d'une notion.

    Retourne 1,0 si la notion n'a pas de prerequis : rien ne bloque son
    enseignement.
    """
    amonts = graphe.get("prerequis", {}).get(cle, [])
    if not amonts:
        return 1.0
    valeurs = [maitrises.get(a, 0.0) for a in amonts]
    return float(np.mean(valeurs)) if valeurs else 1.0
