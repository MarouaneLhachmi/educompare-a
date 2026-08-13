"""
Base de connaissances des referentiels pedagogiques etrangers.
===============================================================

Charge les referentiels depuis `app/data/referentiels/`, qui alimente la base
vectorielle interrogee par l'Agent 5 (Recherche).

Versionnage (plan de transition, phase 1.1)
--------------------------------------------
Un jury demandera d'ou viennent les notions. Tant que la reponse est « d'un
fichier ecrit a la main », toute la chaine en aval — couverture, ecarts,
recommandations, note — est bornee par cette approximation, sans que rien ne
le dise. Le versionnage rend cette limite explicite et datable :

    app/data/referentiels/<CODE_PAYS>/manifeste.json   versions publiees
    app/data/referentiels/<CODE_PAYS>/<version>.json   contenu fige

**L'unite de versionnage est le pays, pas le couple matiere/niveau.** C'est
le programme officiel d'un pays qui est revise, et il l'est pour toutes ses
matieres a la fois : versionner par matiere obligerait a republier cinq
fichiers pour une seule reforme, et a les garder coherents entre eux.

Chaque version declare sa `nature` : `reconstitue` (paraphrase des grandes
lignes, l'etat actuel) ou `officiel` (texte releve a la source, relu par un
humain). Cette distinction est restituee dans le rapport et le back-office —
un resultat produit sur des referentiels reconstitues ne doit pas pouvoir
passer pour un resultat produit sur le texte officiel.

Toute analyse enregistre la signature des versions qu'elle a utilisees. Deux
analyses ne sont comparables que si elles partagent cette signature : sans
cela, une trajectoire compare des pommes et des poires.
"""

import json
import os
import threading

DOSSIER_DONNEES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "referentiels",
)

NATURE_RECONSTITUE = "reconstitue"
NATURE_OFFICIEL = "officiel"

_CACHE: dict[tuple[str, str], dict] = {}
_MANIFESTES: dict[str, dict] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def codes_pays() -> list[str]:
    """Codes des pays presents dans la base, dans l'ordre alphabetique."""
    if not os.path.isdir(DOSSIER_DONNEES):
        return []
    return sorted(
        nom for nom in os.listdir(DOSSIER_DONNEES)
        if os.path.isdir(os.path.join(DOSSIER_DONNEES, nom))
    )


def manifeste(code_pays: str) -> dict:
    """Manifeste d'un pays : identite, version courante, versions publiees."""
    if code_pays in _MANIFESTES:
        return _MANIFESTES[code_pays]
    chemin = os.path.join(DOSSIER_DONNEES, code_pays, "manifeste.json")
    with _LOCK:
        if code_pays not in _MANIFESTES:
            with open(chemin, "r", encoding="utf-8") as fichier:
                _MANIFESTES[code_pays] = json.load(fichier)
    return _MANIFESTES[code_pays]


def version_courante(code_pays: str) -> str:
    """Version publiee en production pour ce pays."""
    return manifeste(code_pays)["version_courante"]


def versions_disponibles(code_pays: str) -> list[dict]:
    return manifeste(code_pays).get("versions", [])


def charger(code_pays: str, version: str | None = None) -> dict:
    """
    Contenu fige d'un referentiel pays, dans une version donnee.

    `version=None` resout vers la version courante declaree au manifeste :
    c'est le comportement de toute la chaine d'analyse, qui n'a aucune raison
    de connaitre les numeros de version.
    """
    version = version or version_courante(code_pays)
    cle_cache = (code_pays, version)
    if cle_cache in _CACHE:
        return _CACHE[cle_cache]

    chemin = os.path.join(DOSSIER_DONNEES, code_pays, f"{version}.json")
    with _LOCK:
        if cle_cache not in _CACHE:
            with open(chemin, "r", encoding="utf-8") as fichier:
                _CACHE[cle_cache] = json.load(fichier)
    return _CACHE[cle_cache]


def _versions_demandees(codes: list[str], versions: dict[str, str] | None) -> dict[str, str]:
    versions = versions or {}
    return {code: versions.get(code) or version_courante(code) for code in codes}


# ---------------------------------------------------------------------------
# Cles « Matiere - Niveau »
# ---------------------------------------------------------------------------

def cles_disponibles() -> list[str]:
    """Liste des couples "Matiere - Niveau" couverts par la base."""
    cles: set[str] = set()
    for code in codes_pays():
        cles.update(charger(code).get("referentiels", {}))
    return sorted(cles)


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


# ---------------------------------------------------------------------------
# Acces au contenu
# ---------------------------------------------------------------------------

def pays_du_referentiel(cle: str, versions: dict[str, str] | None = None) -> list[dict]:
    """
    Referentiels couvrant une cle « Matiere - Niveau », un bloc par pays.

    Conserve la forme historique attendue par les agents et l'interface :
    `{code, pays, drapeau, referentiel, notions}`, enrichie de la version et
    de la nature de la source.
    """
    resultat = []
    for code in codes_pays():
        version = (versions or {}).get(code) or version_courante(code)
        donnees = charger(code, version)
        bloc = (donnees.get("referentiels") or {}).get(cle)
        if not bloc:
            continue
        meta = donnees.get("_meta", {})
        resultat.append(
            {
                "code": code,
                "pays": meta.get("pays", code),
                "drapeau": meta.get("drapeau", ""),
                "referentiel": bloc.get("referentiel", ""),
                "notions": bloc.get("notions", []),
                "version": version,
                "nature": meta.get("nature", NATURE_RECONSTITUE),
            }
        )
    return resultat


def catalogue_pays(cle: str, versions: dict[str, str] | None = None) -> list[dict]:
    """Version allegee (sans les notions) pour l'affichage des selecteurs."""
    return [
        {
            "code": p["code"],
            "pays": p["pays"],
            "drapeau": p.get("drapeau", ""),
            "referentiel": p.get("referentiel", ""),
            "nb_notions": len(p.get("notions", [])),
            "version": p.get("version"),
            "nature": p.get("nature"),
        }
        for p in pays_du_referentiel(cle, versions)
    ]


def notions_a_plat(cle: str, codes: list[str] | None = None,
                   versions: dict[str, str] | None = None) -> list[dict]:
    """
    Aplatit toutes les notions selectionnees sous la forme d'entrees
    indexables dans la base vectorielle :

        {"code", "pays", "drapeau", "referentiel", "notion", "descriptif",
         "texte", "version"}

    `texte` concatene l'intitule et le descriptif : c'est cette chaine qui est
    vectorisee par l'Agent 4.
    """
    entrees = []
    for pays in pays_du_referentiel(cle, versions):
        if codes and pays["code"] not in codes:
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
                    "version": pays.get("version"),
                }
            )
    return entrees


# ---------------------------------------------------------------------------
# Signature de version (comparabilite des analyses)
# ---------------------------------------------------------------------------

def signature_versions(cle: str, codes: list[str] | None = None,
                       versions: dict[str, str] | None = None) -> dict:
    """
    Identifie sans ambiguite le socle de connaissances d'une analyse.

    Persistee sur chaque analyse sous `referentiel_version`. Deux analyses ne
    sont comparables — trajectoire, profils, evolution de la note — que si
    leurs signatures coincident : un durcissement du referentiel deplace les
    scores autant qu'un durcissement du code.
    """
    blocs = [p for p in pays_du_referentiel(cle, versions)
             if not codes or p["code"] in codes]
    par_pays = {p["code"]: p["version"] for p in blocs}
    return {
        "cle": cle,
        "par_pays": par_pays,
        # Chaine courte, comparable d'une analyse a l'autre et lisible dans
        # un journal : « FR:1.0-reconstitue|UK:1.0-reconstitue ».
        "signature": "|".join(f"{code}:{v}" for code, v in sorted(par_pays.items())),
        "natures": sorted({p.get("nature", NATURE_RECONSTITUE) for p in blocs}),
        "entierement_officiel": bool(blocs) and all(
            p.get("nature") == NATURE_OFFICIEL for p in blocs
        ),
    }


# ---------------------------------------------------------------------------
# Statistiques (back-office)
# ---------------------------------------------------------------------------

def statistiques() -> dict:
    """Statistiques de la base de connaissances (dashboard administrateur)."""
    total_notions = 0
    pays_uniques = set()
    for cle in cles_disponibles():
        for pays in pays_du_referentiel(cle):
            pays_uniques.add(pays["pays"])
            total_notions += len(pays.get("notions", []))

    sources = []
    for code in codes_pays():
        meta = charger(code).get("_meta", {})
        sources.append(
            {
                "code": code,
                "pays": meta.get("pays", code),
                "drapeau": meta.get("drapeau", ""),
                "version": meta.get("version"),
                "nature": meta.get("nature", NATURE_RECONSTITUE),
                "source_officielle": meta.get("source_officielle"),
                "relu_par": meta.get("relu_par"),
                "nb_notions": meta.get("nb_notions", 0),
                "nb_versions": len(versions_disponibles(code)),
            }
        )

    return {
        "nb_referentiels": len(cles_disponibles()),
        "nb_pays": len(pays_uniques),
        "nb_notions": total_notions,
        "pays": sorted(pays_uniques),
        "sources": sources,
        "nb_officiels": sum(1 for s in sources if s["nature"] == NATURE_OFFICIEL),
        "nb_reconstitues": sum(1 for s in sources if s["nature"] != NATURE_OFFICIEL),
    }
