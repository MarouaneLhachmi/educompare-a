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


# ---------------------------------------------------------------------------
# Fil d'événements
# ---------------------------------------------------------------------------
#
# `database.journaliser()` enregistre un type technique et un dictionnaire de
# détails. Affichés bruts, ils donnaient une liste de `role_modifie` suivis
# d'un `{'role': 'administrateur'}` — exact, illisible. Cette table traduit
# chaque type en une phrase, un ton et une icône.
#
# Le `ton` sert au filtrage : un administrateur qui ouvre la supervision
# cherche d'abord les incidents, pas les vingt-cinq connexions de la journée.

EVENEMENTS = {
    "connexion": ("→", "Connexion", "neutre"),
    "deconnexion": ("←", "Déconnexion", "neutre"),
    "connexion_atypique": ("⚠", "Connexion atypique détectée", "alerte"),
    "compte_cree": ("✚", "Compte créé", "info"),
    "compte_active": ("✅", "Compte réactivé", "info"),
    "compte_desactive": ("🚫", "Compte désactivé", "alerte"),
    "role_modifie": ("🛡️", "Rôle modifié", "alerte"),
    "analyse_lancee": ("▶", "Analyse lancée", "neutre"),
    "analyse_supprimee": ("🗑", "Analyse supprimée", "alerte"),
    "analyse_supprimee_admin": ("🗑", "Analyse supprimée par un administrateur", "alerte"),
    "programme_cree": ("🎓", "Programme créé", "info"),
    "programme_supprime": ("🗑", "Programme supprimé", "alerte"),
    "retour_enseignant": ("🗳️", "Retour d'enseignant enregistré", "info"),
    "reentrainement_modeles": ("🔄", "Réentraînement des modèles", "info"),
}

# Ce que l'on retient quand un administrateur demande « seulement ce qui
# compte » : tout sauf le va-et-vient ordinaire des connexions et analyses.
TONS_NOTABLES = {"alerte", "info"}


def _resume_details(type_evenement: str, details: dict) -> str:
    """Une ligne lisible à partir du dictionnaire de détails."""
    details = details or {}
    if type_evenement == "role_modifie":
        role = details.get("role")
        return f"nouveau rôle : {'administrateur' if role == 'administrateur' else 'enseignant'}"
    if type_evenement == "connexion_atypique":
        morceaux = [f"risque {details['risque']}/100"] if details.get("risque") else []
        if details.get("methode"):
            morceaux.append(str(details["methode"]))
        return " · ".join(morceaux)
    if type_evenement in ("analyse_lancee",):
        return str(details.get("fichier") or details.get("analyse_id") or "")
    if type_evenement in ("analyse_supprimee", "analyse_supprimee_admin"):
        return f"analyse {details.get('analyse_id', '?')}"
    if type_evenement in ("programme_cree", "programme_supprime"):
        return str(details.get("nom") or details.get("programme_id") or "")
    if type_evenement == "retour_enseignant":
        return str(details.get("type") or "").replace("_", " ")
    if details.get("email"):
        return str(details["email"])
    return " · ".join(f"{k} : {v}" for k, v in details.items() if v is not None)


def mettre_en_forme_evenements(evenements: list[dict], comptes: dict | None = None,
                               notables_seulement: bool = False) -> list[dict]:
    """
    Traduit le journal technique en fil lisible, regroupé par jour.

    `comptes` associe un identifiant d'utilisateur à son nom : un journal qui
    n'affiche que des identifiants opaques oblige à ouvrir un autre écran pour
    savoir de qui l'on parle.
    """
    comptes = comptes or {}
    lignes = []

    for evenement in evenements:
        type_evenement = evenement.get("type", "")
        icone, libelle, ton = EVENEMENTS.get(
            type_evenement, ("•", type_evenement.replace("_", " ").capitalize(), "neutre")
        )
        if notables_seulement and ton not in TONS_NOTABLES:
            continue

        horodatage = evenement.get("horodatage")
        jour = heure = ""
        if hasattr(horodatage, "strftime"):
            jour, heure = horodatage.strftime("%d/%m/%Y"), horodatage.strftime("%H:%M")
        elif horodatage:
            texte = str(horodatage)
            jour, _, heure = texte.partition(" ")
            heure = heure[:5]

        identifiant = evenement.get("utilisateur_id")
        lignes.append({
            "icone": icone,
            "libelle": libelle,
            "ton": ton,
            "jour": jour,
            "heure": heure,
            "acteur": comptes.get(identifiant) or (identifiant or "système"),
            "detail": _resume_details(type_evenement, evenement.get("details")),
        })

    # Regroupement par jour : un fil plat de cinquante lignes ne se lit pas.
    groupes = []
    for ligne in lignes:
        if not groupes or groupes[-1]["jour"] != ligne["jour"]:
            groupes.append({"jour": ligne["jour"], "evenements": []})
        groupes[-1]["evenements"].append(ligne)
    return groupes
