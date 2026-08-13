"""
Base de connaissances des referentiels pedagogiques etrangers.
===============================================================

Charge et met en forme le fichier `app/data/referentiels_etrangers.json`,
qui alimente la base vectorielle interrogee par l'Agent 5 (Recherche).
"""

import json
import os
import threading

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "referentiels_etrangers.json",
)

_CACHE: dict | None = None
_LOCK = threading.Lock()


def _charger_fichier() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _LOCK:
        if _CACHE is None:
            with open(DATA_PATH, "r", encoding="utf-8") as fichier:
                _CACHE = json.load(fichier)
    return _CACHE


def cles_disponibles() -> list[str]:
    """Liste des couples "Matiere - Niveau" couverts par la base."""
    return [cle for cle in _charger_fichier() if not cle.startswith("_")]


def matieres() -> list[str]:
    return sorted({cle.split(" - ", 1)[0] for cle in cles_disponibles()})


def niveaux(matiere: str | None = None) -> list[str]:
    resultat = set()
    for cle in cles_disponibles():
        m, _, n = cle.partition(" - ")
        if matiere is None or m == matiere:
            resultat.add(n)
    return sorted(resultat)


def cle_pour(matiere: str, niveau: str) -> str:
    """Cle exacte si elle existe, sinon repli sur la premiere cle disponible."""
    cle = f"{matiere} - {niveau}"
    disponibles = cles_disponibles()
    if cle in disponibles:
        return cle
    # Repli tolerant : meme matiere, niveau different
    for candidate in disponibles:
        if candidate.startswith(f"{matiere} - "):
            return candidate
    return disponibles[0]


def pays_du_referentiel(cle: str) -> list[dict]:
    """Retourne la liste des pays/referentiels couverts pour une cle donnee."""
    return _charger_fichier().get(cle, [])


def catalogue_pays(cle: str) -> list[dict]:
    """Version allegee (sans les notions) pour l'affichage des selecteurs."""
    return [
        {
            "code": p["code"],
            "pays": p["pays"],
            "drapeau": p.get("drapeau", ""),
            "referentiel": p.get("referentiel", ""),
            "nb_notions": len(p.get("notions", [])),
        }
        for p in pays_du_referentiel(cle)
    ]


def notions_a_plat(cle: str, codes_pays: list[str] | None = None) -> list[dict]:
    """
    Aplatit toutes les notions selectionnees sous la forme d'entrees
    indexables dans la base vectorielle :

        {"code", "pays", "drapeau", "referentiel", "notion", "descriptif", "texte"}

    `texte` concatene l'intitule et le descriptif : c'est cette chaine qui est
    vectorisee par l'Agent 4.
    """
    entrees = []
    for pays in pays_du_referentiel(cle):
        if codes_pays and pays["code"] not in codes_pays:
            continue
        for notion in pays.get("notions", []):
            intitule = notion["intitule"]
            descriptif = notion.get("descriptif", "")
            entrees.append(
                {
                    "code": pays["code"],
                    "pays": pays["pays"],
                    "drapeau": pays.get("drapeau", ""),
                    "referentiel": pays.get("referentiel", ""),
                    "notion": intitule,
                    "descriptif": descriptif,
                    "texte": f"{intitule}. {descriptif}".strip(),
                }
            )
    return entrees


def statistiques() -> dict:
    """Statistiques de la base de connaissances (dashboard administrateur)."""
    total_notions = 0
    pays_uniques = set()
    for cle in cles_disponibles():
        for pays in pays_du_referentiel(cle):
            pays_uniques.add(pays["pays"])
            total_notions += len(pays.get("notions", []))
    return {
        "nb_referentiels": len(cles_disponibles()),
        "nb_pays": len(pays_uniques),
        "nb_notions": total_notions,
        "pays": sorted(pays_uniques),
    }
