"""
Aides de présentation partagées par les blueprints.
====================================================

Tri et pagination ne sont pas de la logique métier : ils ne changent ni ce que
le système mesure, ni ce qu'il décide. Ils déterminent l'ordre et la quantité
de ce qui est montré. Ils vivent donc dans la couche routes, à côté des vues
qui les utilisent, et non dans les modules fonctionnels.

Les deux historiques — celui de l'enseignant et celui du back-office — rendent
le même gabarit ; sans ces fonctions communes, ils divergeraient.
"""

TRIS_ANALYSES = {
    "recent": ("Plus récentes d'abord", "date_creation_iso", True),
    "ancien": ("Plus anciennes d'abord", "date_creation_iso", False),
    "note_desc": ("Meilleure note d'abord", "resume_note_globale", True),
    "note_asc": ("Note la plus faible d'abord", "resume_note_globale", False),
    "nom": ("Nom du document (A → Z)", "titre_cours", False),
}

TRI_PAR_DEFAUT = "recent"

# Au-delà, une page devient illisible et lourde à rendre. La valeur vaut pour
# le back-office comme pour l'espace enseignant : un enseignant prolifique
# rencontre le même mur qu'un administrateur.
TAILLE_PAGE = 12


def trier_analyses(analyses: list[dict], tri: str) -> list[dict]:
    """Ordonne une liste d'analyses selon une clé de tri déclarée."""
    if tri not in TRIS_ANALYSES:
        tri = TRI_PAR_DEFAUT
    _, champ, decroissant = TRIS_ANALYSES[tri]

    def cle(analyse: dict):
        valeur = analyse.get(champ)
        if champ == "resume_note_globale":
            try:
                return float(valeur or 0)
            except (TypeError, ValueError):
                return 0.0
        # Les analyses sans titre ni date ne doivent pas s'intercaler au
        # hasard : elles sont repoussées en fin de liste dans les deux sens.
        return str(valeur or "").lower()

    return sorted(analyses, key=cle, reverse=decroissant)


def paginer(elements: list, page: int, taille: int = TAILLE_PAGE) -> dict:
    """
    Découpe une liste pour l'affichage.

    Retourne les éléments de la page demandée et de quoi construire la
    navigation, y compris quand la page demandée n'existe pas — un numéro de
    page venu d'une URL modifiée à la main ne doit pas produire une page vide
    sans explication.
    """
    total = len(elements)
    nb_pages = max(1, -(-total // taille))  # division entière par excès
    page = max(1, min(int(page or 1), nb_pages))
    debut = (page - 1) * taille

    return {
        "elements": elements[debut : debut + taille],
        "page": page,
        "nb_pages": nb_pages,
        "total": total,
        "taille": taille,
        "premier": debut + 1 if total else 0,
        "dernier": min(debut + taille, total),
        "a_precedent": page > 1,
        "a_suivant": page < nb_pages,
        "paginee": nb_pages > 1,
    }


def page_demandee(args) -> int:
    """Numéro de page issu de la requête, tolérant à une valeur aberrante."""
    try:
        return max(1, int(args.get("page", 1)))
    except (TypeError, ValueError):
        return 1
