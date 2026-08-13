"""
Entrainement local des modeles des modules fonctionnels.
=========================================================

Les neuf agents s'appuient sur des modeles pre-entraines telecharges depuis
HuggingFace ou sur des artefacts produits en Colab. Les cinq modules
fonctionnels, eux, doivent apprendre **sur les donnees de cette instance** :
le corpus de referentiels charge, les analyses deja realisees, l'historique
des connexions. Aucun modele de langage n'intervient ici.

Contrainte structurante
-----------------------

Une instance qui demarre ne dispose de presque aucune donnee : quinze
analyses, deux comptes, trente-quatre evenements au moment de la conception.
Cela interdit l'apprentissage supervise sur les donnees de production et
oriente l'ensemble des choix :

- on privilegie les methodes **non supervisees** (detection d'anomalies,
  regroupement, graphes), qui n'exigent aucune etiquette ;
- quand une etiquette est indispensable, on exploite le seul corpus
  reellement etiquete dont on dispose : le fichier de referentiels, ou
  chaque notion porte sa matiere et son niveau ;
- les modeles qui ont besoin de volume (prediction de duree) demarrent sur
  un jeu synthetique explicitement documente, puis sont **reentraines sur les
  donnees reelles** des qu'elles sont assez nombreuses.

Chaque modele expose sa provenance (`source_donnees`) et le nombre
d'observations sur lequel il a ete ajuste, afin que le back-office puisse
distinguer un modele credible d'un modele encore en amorcage.

Cache
-----

Les modeles sont mis en cache en memoire pour la duree du processus, et sur
disque dans `app/models/` pour survivre a un redemarrage. Un modele dont les
donnees d'entrainement ont change est reentraine a la demande par le module
de supervision.
"""

import os
import threading
import time

import numpy as np

from app.config import Config
from app.services import referentiels

# Nombre minimal d'observations reelles au-dela duquel un modele cesse de
# fonctionner en amorcage et devient exploitable.
SEUIL_DONNEES_REELLES = 30

_CACHE: dict[str, dict] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Corpus disponibles
# ---------------------------------------------------------------------------

def corpus_referentiels() -> dict:
    """
    Corpus etiquete extrait du fichier de referentiels.

    C'est la seule source d'etiquettes fiables de l'instance : chaque notion
    y est rattachee a une matiere, un niveau et un pays. Elle sert de base
    d'apprentissage au triage documentaire et a la prediction de matiere.
    """
    textes, matieres, niveaux, cles = [], [], [], []
    for cle in referentiels.cles_disponibles():
        matiere, _, niveau = cle.partition(" - ")
        for notion in referentiels.notions_a_plat(cle):
            textes.append(notion["texte"])
            matieres.append(matiere)
            niveaux.append(niveau)
            cles.append(cle)
    return {
        "textes": textes,
        "matieres": matieres,
        "niveaux": niveaux,
        "cles": cles,
        "nb": len(textes),
        "matieres_distinctes": sorted(set(matieres)),
    }


def corpus_analyses(champs: list[str] | None = None) -> list[dict]:
    """
    Analyses terminees deja persistees, source d'apprentissage des modeles
    qui portent sur l'execution (duree, profils d'indicateurs).
    """
    from app.services import database

    analyses = [
        a for a in database.lister_analyses()
        if a.get("statut") == "TERMINEE"
    ]
    if champs is None:
        return analyses
    return [{c: a.get(c) for c in champs} for a in analyses]


def corpus_evenements(types: set[str] | None = None) -> list[dict]:
    """Journal d'evenements, source d'apprentissage du profil de connexion."""
    from app.services import database

    evenements = database.lister_evenements(2000)
    if types:
        evenements = [e for e in evenements if e.get("type") in types]
    return evenements


# ---------------------------------------------------------------------------
# Encodage du temps
# ---------------------------------------------------------------------------

def encoder_horaire(horodatage) -> tuple[float, float, float, float]:
    """
    Encode un instant en coordonnees cycliques.

    Une heure representee par un entier de 0 a 23 place minuit et 23 h aux
    deux extremites de l'echelle, alors qu'ils sont voisins dans le temps.
    La projection sur un cercle (sinus, cosinus) retablit cette proximite —
    indispensable des lors qu'un modele calcule des distances.
    """
    import datetime

    if isinstance(horodatage, (int, float)):
        instant = datetime.datetime.fromtimestamp(horodatage)
    elif isinstance(horodatage, datetime.datetime):
        instant = horodatage
    else:
        try:
            instant = datetime.datetime.fromisoformat(str(horodatage))
        except ValueError:
            instant = datetime.datetime.now()

    angle_heure = 2 * np.pi * (instant.hour + instant.minute / 60) / 24
    angle_jour = 2 * np.pi * instant.weekday() / 7
    return (
        float(np.sin(angle_heure)), float(np.cos(angle_heure)),
        float(np.sin(angle_jour)), float(np.cos(angle_jour)),
    )


# ---------------------------------------------------------------------------
# Cache des modeles
# ---------------------------------------------------------------------------

def _chemin_cache(nom: str) -> str:
    os.makedirs(Config.MODELS_FOLDER, exist_ok=True)
    return os.path.join(Config.MODELS_FOLDER, f"{nom}.joblib")


def obtenir(nom: str, fabrique, forcer: bool = False) -> dict:
    """
    Retourne le modele `nom`, en l'entrainant au premier appel.

    `fabrique` est une fonction sans argument retournant un dictionnaire
    contenant au minimum les cles `modele`, `source_donnees` et
    `nb_observations`. Le resultat est mis en cache memoire et disque.

    Toute defaillance d'entrainement est capturee : l'appelant recoit alors
    un dictionnaire `{"modele": None, "erreur": ...}` et bascule sur son
    repli, exactement comme pour les modeles des agents.
    """
    if not forcer and nom in _CACHE:
        return _CACHE[nom]

    with _LOCK:
        if not forcer and nom in _CACHE:
            return _CACHE[nom]

        chemin = _chemin_cache(nom)
        if not forcer and os.path.exists(chemin):
            try:
                import joblib

                enregistre = joblib.load(chemin)
                if isinstance(enregistre, dict) and enregistre.get("modele") is not None:
                    enregistre["origine"] = "cache disque"
                    _CACHE[nom] = enregistre
                    return enregistre
            except Exception:
                pass  # cache illisible ou incompatible : on reentraine

        debut = time.time()
        try:
            resultat = fabrique() or {}
            resultat.setdefault("modele", None)
            resultat.setdefault("source_donnees", "inconnue")
            resultat.setdefault("nb_observations", 0)
            resultat["erreur"] = None
        except Exception as exc:
            resultat = {
                "modele": None, "source_donnees": "indisponible",
                "nb_observations": 0, "erreur": str(exc)[:200],
            }

        resultat["nom"] = nom
        resultat["duree_entrainement_s"] = round(time.time() - debut, 2)
        resultat["horodatage_entrainement"] = time.time()
        resultat["origine"] = "entrainement"
        resultat["amorcage"] = resultat["nb_observations"] < SEUIL_DONNEES_REELLES

        if resultat["modele"] is not None:
            try:
                import joblib

                joblib.dump(resultat, chemin)
            except Exception:
                pass  # l'absence de cache disque n'empeche rien

        _CACHE[nom] = resultat

    return _CACHE[nom]


def invalider(nom: str | None = None) -> list[str]:
    """
    Vide le cache d'un modele (ou de tous) et supprime son artefact disque,
    afin que le prochain appel le reentraine. Utilise par le module de
    supervision lorsqu'une derive est detectee.
    """
    with _LOCK:
        noms = [nom] if nom else list(_CACHE.keys())
        for cle in noms:
            _CACHE.pop(cle, None)
            chemin = _chemin_cache(cle)
            if os.path.exists(chemin):
                try:
                    os.remove(chemin)
                except OSError:
                    pass
    return noms


def etat_modeles() -> list[dict]:
    """Etat des modeles de modules deja entraines, pour la supervision."""
    with _LOCK:
        entrees = list(_CACHE.items())
    return [
        {
            "nom": nom,
            "actif": donnees.get("modele") is not None,
            "source_donnees": donnees.get("source_donnees"),
            "nb_observations": donnees.get("nb_observations", 0),
            "amorcage": donnees.get("amorcage", True),
            "origine": donnees.get("origine"),
            "duree_entrainement_s": donnees.get("duree_entrainement_s"),
            "age_heures": round(
                (time.time() - donnees.get("horodatage_entrainement", time.time())) / 3600, 1
            ),
            "erreur": donnees.get("erreur"),
            **{k: v for k, v in donnees.items() if k.startswith("metrique_")},
        }
        for nom, donnees in entrees
    ]


# ---------------------------------------------------------------------------
# Vectorisation partagee des textes
# ---------------------------------------------------------------------------

def encoder_textes(textes: list[str]) -> tuple[np.ndarray | None, str]:
    """
    Encode une liste de textes dans l'espace vectoriel de l'Agent 4.

    Reutiliser le meme encodeur que le pipeline d'analyse evite d'embarquer
    un second modele et garantit que les modules et les agents raisonnent
    dans le meme espace. Repli TF-IDF si le modele neuronal est absent.
    """
    if not textes:
        return None, "aucun texte"

    from app.agents import agent4_vectorisation

    modele = agent4_vectorisation.charger_modele_transformer()
    if modele is not None:
        vecteurs = agent4_vectorisation.VectoriseurTransformer(modele).encoder(textes)
        return np.asarray(vecteurs, dtype="float64"), "sentence-transformers"

    try:
        vectoriseur = agent4_vectorisation.VectoriseurLSA(textes)
        return np.asarray(vectoriseur.encoder(textes), dtype="float64"), "LSA (TF-IDF + SVD)"
    except Exception as exc:
        return None, f"indisponible : {exc}"
