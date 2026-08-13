"""
Registre des modeles d'apprentissage.
======================================

Le systeme mobilise plusieurs modeles de nature differente :

- des modeles **telecharges depuis HuggingFace** (bi-encodeur, cross-encodeur) ;
- des modeles **entraines hors ligne** dans les notebooks Colab du dossier
  `notebooks/`, puis deposes sous forme d'artefacts dans `app/models/`.

Ce module centralise leur cycle de vie et impose trois invariants a l'echelle
de toute l'application :

1. **Chargement paresseux et unique.** Un modele n'est charge qu'au premier
   usage reel, puis conserve en memoire pour le reste du processus. Une
   analyse qui n'a pas besoin du cross-encodeur ne paie pas son chargement.

2. **Aucun modele n'est bloquant.** Toute absence (artefact non depose, poids
   corrompus, pas de reseau au premier telechargement) est capturee, tracee,
   et l'agent appelant bascule sur son repli. C'est la meme discipline que
   celle appliquee aux agents generatifs.

3. **Etat observable.** `etat()` alimente la supervision technique du
   back-office : l'administrateur voit en un coup d'oeil quels modeles sont
   reellement actifs et lesquels tournent en mode degrade.
"""

import os
import threading
import time

from app.config import Config

# ---------------------------------------------------------------------------
# Declaration des modeles connus du systeme
# ---------------------------------------------------------------------------
# `source` :
#   "huggingface" -> telecharge et mis en cache par la bibliotheque
#   "artefact"    -> fichier produit par un notebook, depose dans MODELS_FOLDER
CATALOGUE_MODELES = {
    "cross_encoder": {
        "libelle": "Cross-encodeur de re-ranking",
        "agent": "Agent 6 — Comparaison",
        "type": "Deep Learning",
        "source": "huggingface",
        "reference": Config.CROSS_ENCODER_MODEL,
        "role": "Re-score conjointement la paire (unité de cours, notion) pour écarter les faux positifs du bi-encodeur.",
        "repli": "Similarité cosinus du bi-encodeur seule.",
    },
    "couverture_clf": {
        "libelle": "Classifieur de couverture calibré",
        "agent": "Agent 6 — Comparaison",
        "type": "Machine Learning",
        "source": "artefact",
        "fichier": "couverture_clf.joblib",
        "role": "Fusionne les signaux de similarité en une probabilité de couverture calibrée.",
        "repli": "Seuils fixes calibrés par moteur de vectorisation.",
    },
    "bloom_clf": {
        "libelle": "Classifieur de taxonomie de Bloom",
        "agent": "Agent 7 — Évaluation",
        "type": "Deep Learning",
        "source": "artefact",
        "fichier": "bloom_clf.joblib",
        "role": "Situe chaque objectif et chaque exercice sur les six niveaux cognitifs de Bloom.",
        "repli": "Heuristique par verbes d'action.",
    },
    "niveau_reg": {
        "libelle": "Estimateur neuronal de niveau scolaire",
        "agent": "Agent 7 — Évaluation",
        "type": "Deep Learning",
        "source": "artefact",
        "fichier": "niveau_reg.joblib",
        "role": "Estime le niveau scolaire réel de chaque unité de contenu, et donc la progression de difficulté.",
        "repli": "Indice de lisibilité statistique (longueur des phrases, rareté lexicale).",
    },
    "dkt_lstm": {
        "libelle": "Deep Knowledge Tracing (LSTM)",
        "agent": "Agent 8 — Recommandations",
        "type": "Deep Learning",
        "source": "artefact",
        "fichier": "dkt_lstm.pt",
        "role": "Modélise l'acquisition des notions au fil des séances pour simuler la progression de l'élève.",
        "repli": "Bayesian Knowledge Tracing paramétré.",
    },
    "dqn_planificateur": {
        "libelle": "Deep Q-Network de planification",
        "agent": "Agent 8 — Recommandations",
        "type": "Apprentissage par renforcement profond",
        "source": "artefact",
        "fichier": "dqn_planificateur.pt",
        "role": "Choisit la séquence d'interventions pédagogiques maximisant la maîtrise finale.",
        "repli": "Q-Learning tabulaire entraîné localement, puis priorisation déterministe.",
    },
}

_CACHE: dict[str, object] = {}
_ETATS: dict[str, dict] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def chemin_artefact(nom: str) -> str | None:
    """Chemin attendu de l'artefact d'un modele entraine hors ligne."""
    declaration = CATALOGUE_MODELES.get(nom)
    if not declaration or declaration["source"] != "artefact":
        return None
    return os.path.join(Config.MODELS_FOLDER, declaration["fichier"])


def artefact_present(nom: str) -> bool:
    chemin = chemin_artefact(nom)
    return bool(chemin and os.path.exists(chemin))


def _enregistrer_etat(nom: str, statut: str, detail: str, duree_ms: int = 0) -> None:
    _ETATS[nom] = {
        "statut": statut,  # "actif" | "absent" | "erreur" | "desactive"
        "detail": detail,
        "duree_chargement_ms": duree_ms,
        "horodatage": time.time(),
    }


# ---------------------------------------------------------------------------
# Chargeurs specifiques
# ---------------------------------------------------------------------------

def _charger_cross_encoder():
    if not Config.USE_CROSS_ENCODER:
        _enregistrer_etat("cross_encoder", "desactive",
                          "Désactivé par configuration (USE_CROSS_ENCODER=false).")
        return None
    from sentence_transformers import CrossEncoder

    modele = CrossEncoder(Config.CROSS_ENCODER_MODEL, max_length=384)
    return modele


def _charger_joblib(nom: str):
    chemin = chemin_artefact(nom)
    if not chemin or not os.path.exists(chemin):
        _enregistrer_etat(
            nom, "absent",
            f"Artefact non déposé ({CATALOGUE_MODELES[nom]['fichier']}) — repli actif.",
        )
        return None
    import joblib

    return joblib.load(chemin)


def _charger_torch(nom: str):
    chemin = chemin_artefact(nom)
    if not chemin or not os.path.exists(chemin):
        _enregistrer_etat(
            nom, "absent",
            f"Artefact non déposé ({CATALOGUE_MODELES[nom]['fichier']}) — repli actif.",
        )
        return None
    import torch

    # `weights_only=False` est necessaire : les notebooks sauvegardent un
    # dictionnaire contenant a la fois les poids et les metadonnees
    # (dimensions, mapping des notions, hyperparametres d'entrainement).
    return torch.load(chemin, map_location="cpu", weights_only=False)


_CHARGEURS = {
    "cross_encoder": _charger_cross_encoder,
    "couverture_clf": lambda: _charger_joblib("couverture_clf"),
    "bloom_clf": lambda: _charger_joblib("bloom_clf"),
    "niveau_reg": lambda: _charger_joblib("niveau_reg"),
    "dkt_lstm": lambda: _charger_torch("dkt_lstm"),
    "dqn_planificateur": lambda: _charger_torch("dqn_planificateur"),
}


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def charger(nom: str):
    """
    Retourne le modele demande, ou `None` s'il n'est pas disponible.

    L'appelant DOIT traiter le cas `None` : c'est le contrat qui garantit
    qu'aucun modele n'est bloquant pour le pipeline.
    """
    if nom not in CATALOGUE_MODELES:
        raise KeyError(f"Modèle inconnu du registre : {nom}")

    if nom in _CACHE:
        return _CACHE[nom]

    with _LOCK:
        if nom in _CACHE:
            return _CACHE[nom]

        debut = time.time()
        try:
            modele = _CHARGEURS[nom]()
        except Exception as exc:
            modele = None
            _enregistrer_etat(nom, "erreur", f"Chargement impossible : {exc}"[:220])

        duree = int((time.time() - debut) * 1000)
        if modele is not None:
            declaration = CATALOGUE_MODELES[nom]
            origine = (
                declaration.get("reference")
                if declaration["source"] == "huggingface"
                else declaration.get("fichier")
            )
            _enregistrer_etat(nom, "actif", f"Chargé depuis {origine}.", duree)

        # On memorise meme la valeur None : un modele absent ne doit pas etre
        # recherche a chaque analyse (le disque serait sollicite inutilement).
        _CACHE[nom] = modele

    return _CACHE[nom]


def est_disponible(nom: str) -> bool:
    return charger(nom) is not None


def reinitialiser(nom: str | None = None) -> None:
    """
    Vide le cache : utile apres avoir depose un nouvel artefact issu d'un
    notebook, sans avoir a redemarrer l'application.
    """
    with _LOCK:
        if nom is None:
            _CACHE.clear()
            _ETATS.clear()
        else:
            _CACHE.pop(nom, None)
            _ETATS.pop(nom, None)


def etat() -> list[dict]:
    """
    Etat de chaque modele du catalogue, pour la supervision technique.

    N'declenche aucun chargement : un modele jamais sollicite est signale
    comme « non sollicité », ce qui est une information en soi.
    """
    lignes = []
    for nom, declaration in CATALOGUE_MODELES.items():
        connu = _ETATS.get(nom)
        if connu is None:
            if declaration["source"] == "artefact":
                statut = "disponible" if artefact_present(nom) else "absent"
                detail = (
                    "Artefact présent, pas encore sollicité."
                    if statut == "disponible"
                    else f"Artefact non déposé ({declaration['fichier']}) — repli actif."
                )
            else:
                statut = "non_sollicite"
                detail = "Sera téléchargé et mis en cache au premier usage."
            connu = {"statut": statut, "detail": detail, "duree_chargement_ms": 0}

        lignes.append(
            {
                "cle": nom,
                "libelle": declaration["libelle"],
                "agent": declaration["agent"],
                "type": declaration["type"],
                "source": declaration["source"],
                "role": declaration["role"],
                "repli": declaration["repli"],
                "chemin_attendu": declaration.get("fichier"),
                **connu,
            }
        )
    return lignes


def resume() -> dict:
    """Synthese chiffree pour le tableau de bord administrateur."""
    lignes = etat()
    actifs = sum(1 for l in lignes if l["statut"] == "actif")
    disponibles = sum(1 for l in lignes if l["statut"] in {"actif", "disponible"})
    return {
        "nb_modeles": len(lignes),
        "nb_actifs": actifs,
        "nb_disponibles": disponibles,
        "nb_en_repli": sum(1 for l in lignes if l["statut"] in {"absent", "erreur"}),
        "dossier_artefacts": Config.MODELS_FOLDER,
    }
