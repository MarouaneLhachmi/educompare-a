"""
Mesure des intervalles d'ancrage du corpus de reference.
=========================================================

Un ancrage n'est pas une valeur decretee : c'est une valeur **mesuree** sur le
comportement actuel du systeme, entouree d'une marge de tolerance. Les tests
d'ancrage verifient ensuite que le systeme reste dans cette bande.

C'est precisement l'outil qui manquait lors du durcissement de l'Agent 6 : la
note du cours de demonstration est passee de 74,9 a 54 — le resultat recherche,
mais rien dans le systeme ne l'aurait signale si ca n'avait pas ete le cas.

Usage :
    python tests/mesurer_ancrages.py             # mesure et ecrit ancrages.json
    python tests/mesurer_ancrages.py --afficher  # mesure sans ecrire

Apres un changement de seuil VOULU, relancer ce script et **commiter la
difference** : le diff de `ancrages.json` devient la trace explicite de
l'impact du changement sur chaque document du corpus.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

# Neutralisation du modele de langage : la mesure doit etre reproductible et
# hors ligne, comme les tests qui l'exploiteront.
os.environ["GEMINI_API_KEY"] = ""

from app.config import Config  # noqa: E402
from app.services import database, gemini_client  # noqa: E402

DOSSIER_TESTS = os.path.dirname(os.path.abspath(__file__))
CATALOGUE_PATH = os.path.join(DOSSIER_TESTS, "corpus_reference", "catalogue.json")
ANCRAGES_PATH = os.path.join(DOSSIER_TESTS, "ancrages.json")

# Marges de tolerance, en points. Assez larges pour absorber le bruit d'une
# machine a l'autre, assez etroites pour qu'un deplacement de seuil sorte de
# la bande : le durcissement de l'Agent 6 avait deplace la note de 21 points.
MARGE_NOTE = 8.0
MARGE_COUVERTURE = 10.0


def _neutraliser_llm() -> None:
    def _indisponible(*args, **kwargs):
        raise gemini_client.LLMUnavailable("Modele de langage neutralise (mesure).")

    gemini_client.generate_text = _indisponible
    gemini_client.generate_json = _indisponible
    gemini_client.is_configured = lambda: False


def _forcer_base_memoire() -> None:
    database._STATE.update(
        {"client": None, "db": database.InMemoryDB(), "mode": "memoire",
         "erreur": "base en memoire (mesure)"}
    )


def _environnement(analyse: dict) -> dict:
    """
    Contexte de la mesure. Deux ancrages ne sont comparables que s'ils ont ete
    produits dans le meme environnement : le repli LSA et le modele neuronal
    ne donnent pas les memes scores, et le dire evite un faux diagnostic de
    regression.
    """
    agent4 = analyse.get("agent4") or {}
    agent5 = analyse.get("agent5") or {}
    agent6 = analyse.get("agent6") or {}
    return {
        "moteur_vectorisation": agent4.get("moteur"),
        "type_moteur": agent4.get("type_moteur"),
        "repli_vectorisation_actif": bool(agent4.get("repli_actif")),
        "dimension": agent4.get("dimension"),
        "moteur_index": (agent5.get("index") or {}).get("moteur"),
        "cross_encodeur_demande": Config.USE_CROSS_ENCODER,
        "cross_encodeur_applique": bool((agent6.get("reranking") or {}).get("applique")),
        "source_decision": (agent6.get("decision") or {}).get("source"),
        "seuils": (agent6.get("decision") or {}).get("seuils"),
    }


def mesurer_document(document: dict) -> dict | None:
    """Execute la chaine complete sur un document et resume ce qui est ancre."""
    from app.modules import module_traitement_analyse

    chemin = os.path.join(DOSSIER_TESTS, "corpus_reference", document["fichier"])
    analyse = module_traitement_analyse.run_analysis(
        pdf_path=chemin,
        matiere=document["matiere"],
        niveau=document["niveau"],
        nom_fichier_original=document["fichier"],
    )

    if analyse["statut"] == "ECHEC":
        return {
            "fichier": document["fichier"],
            "nature": document["nature"],
            "statut": "ECHEC",
            "erreur": str(analyse.get("erreur"))[:200],
        }

    agent6 = analyse.get("agent6") or {}
    agent7 = analyse.get("agent7") or {}
    agent8 = analyse.get("agent8") or {}

    note = float(agent7.get("note_globale") or 0)
    couverture = float(agent6.get("score_global_pct") or 0)

    return {
        "fichier": document["fichier"],
        "nature": document["nature"],
        "statut": "TERMINEE",
        "mesure": {
            "note_globale": round(note, 1),
            "couverture_pct": round(couverture, 1),
            "nb_notions_manquantes": agent6.get("nb_notions_manquantes"),
            "nb_unites": (analyse.get("agent3") or {}).get("nb_unites"),
            "nb_chapitres": (analyse.get("agent1") or {}).get("nb_chapitres"),
            "nb_recommandations": len(agent8.get("recommandations") or []),
            "niveau_maturite": agent7.get("niveau_maturite"),
        },
        "intervalle": {
            "note_globale": [round(max(0.0, note - MARGE_NOTE), 1),
                             round(min(100.0, note + MARGE_NOTE), 1)],
            "couverture_pct": [round(max(0.0, couverture - MARGE_COUVERTURE), 1),
                               round(min(100.0, couverture + MARGE_COUVERTURE), 1)],
        },
        "environnement": _environnement(analyse),
    }


def mesurer_tout() -> dict:
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as fichier:
        catalogue = json.load(fichier)

    documents = {}
    for document in catalogue:
        print(f"  {document['fichier']:<28} ", end="", flush=True)
        try:
            resultat = mesurer_document(document)
        except Exception as exc:
            resultat = {
                "fichier": document["fichier"], "nature": document["nature"],
                "statut": "ECHEC", "erreur": f"{type(exc).__name__}: {exc}"[:200],
            }
        documents[document["fichier"]] = resultat

        if resultat["statut"] == "ECHEC":
            print(f"ECHEC — {resultat['erreur'][:60]}")
        else:
            mesure = resultat["mesure"]
            print(f"note {mesure['note_globale']:>5}  "
                  f"couverture {mesure['couverture_pct']:>5} %  "
                  f"({mesure['nb_notions_manquantes']} notions manquantes)")

    return {
        "_meta": {
            "description": (
                "Intervalles d'ancrage mesures sur le corpus de reference. "
                "Regenerer avec `python tests/mesurer_ancrages.py` apres un "
                "changement volontaire de seuil, et commiter la difference."
            ),
            "mesure_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "marge_note": MARGE_NOTE,
            "marge_couverture": MARGE_COUVERTURE,
            "modele_de_langage": "neutralise (mode degrade)",
        },
        "documents": documents,
    }


def main() -> int:
    parseur = argparse.ArgumentParser(description="Mesure des ancrages de non-regression")
    parseur.add_argument("--afficher", action="store_true",
                         help="affiche les mesures sans ecrire ancrages.json")
    args = parseur.parse_args()

    _neutraliser_llm()
    _forcer_base_memoire()

    print("Mesure des ancrages sur le corpus de reference "
          "(modele de langage neutralise) :\n")
    ancrages = mesurer_tout()

    if args.afficher:
        print("\n(--afficher : ancrages.json non modifié)")
        return 0

    with open(ANCRAGES_PATH, "w", encoding="utf-8") as fichier:
        json.dump(ancrages, fichier, ensure_ascii=False, indent=2)
    print(f"\nAncrages écrits dans {ANCRAGES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
