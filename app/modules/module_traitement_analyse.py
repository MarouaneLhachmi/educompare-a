"""
Module Traitement et Analyse
=============================

Role (rapport de conception, section 3.3.1) : coeur fonctionnel du systeme.
Ce module coordonne l'execution sequentielle des neuf agents d'intelligence
artificielle, gere les erreurs susceptibles de survenir au cours du
traitement, et agrege les resultats intermediaires produits par chaque agent.

Il ajoute trois responsabilites indispensables a une execution reelle :

- **Execution asynchrone** : l'analyse complete (9 agents, dont plusieurs
  appels a un modele de langage) prend plusieurs dizaines de secondes. Elle
  est donc lancee dans un fil d'execution dedie, et l'interface interroge
  l'avancement en temps reel.
- **Suivi d'avancement** : chaque agent publie son etat (en attente, en cours,
  termine, replie, en echec) et sa duree d'execution.
- **Tolerance aux pannes** : un agent non critique qui echoue n'interrompt pas
  la chaine ; son repli est active et l'incident est trace dans le journal
  d'execution restitue a l'utilisateur.
"""

import datetime
import threading
import time
import traceback
import uuid

from app.agents import (
    agent1_extraction,
    agent2_comprehension,
    agent3_decoupage,
    agent4_vectorisation,
    agent5_recherche,
    agent6_comparaison,
    agent7_evaluation,
    agent8_recommandations,
    agent9_rapport,
)
from app.agents import CATALOGUE
from app.services import database, prediction_execution, referentiels

# Registre d'avancement des analyses en cours (memoire du processus).
PROGRESSION: dict[str, dict] = {}
_LOCK = threading.Lock()

# Repartition de secours, utilisee si la prediction de duree est
# indisponible. Ce ne sont plus les poids de reference : ils viennent
# desormais du modele de `services/prediction_execution`, ajuste sur les
# caracteristiques du document analyse.
POIDS_AGENTS = dict(prediction_execution.REPARTITION_APRIORI)

# Agents sans lesquels l'analyse n'a plus de sens.
AGENTS_CRITIQUES = {"agent1", "agent3", "agent4", "agent5", "agent6"}


# ---------------------------------------------------------------------------
# Suivi d'avancement
# ---------------------------------------------------------------------------

def _initialiser_progression(analyse_id: str, nom_fichier: str,
                             prediction: dict | None = None) -> dict:
    prediction = prediction or {}
    etat = {
        "analyse_id": analyse_id,
        "nom_fichier": nom_fichier,
        "statut": "EN_COURS",
        "pourcentage": 0,
        "agent_courant": None,
        "message": "Initialisation du pipeline…",
        "debut": time.time(),
        "duree_totale_s": None,
        # Durees predites par le modele : elles remplacent les poids fixes et
        # alimentent a la fois la barre de progression et le temps restant.
        "poids": prediction.get("poids") or {
            a: 100 * p for a, p in POIDS_AGENTS.items()
        },
        "durees_prevues": prediction.get("durees") or {},
        "duree_totale_prevue_s": prediction.get("duree_totale_prevue_s"),
        "temps_restant_s": prediction.get("duree_totale_prevue_s"),
        "prediction_source": prediction.get("source"),
        "prediction_amorcage": prediction.get("amorcage", True),
        "erreur": None,
        "agents": [
            {
                "cle": agent["cle"],
                "numero": agent["numero"],
                "nom": agent["nom"],
                "nature": agent["nature"],
                "icone": agent["icone"],
                "statut": "en_attente",
                "duree_s": None,
                "detail": None,
            }
            for agent in CATALOGUE
        ],
        "journal": [],
    }
    with _LOCK:
        PROGRESSION[analyse_id] = etat
    return etat


def _maj_agent(analyse_id: str, cle: str, statut: str, detail: str | None = None,
                     duree: float | None = None) -> None:
    with _LOCK:
        etat = PROGRESSION.get(analyse_id)
        if not etat:
            return
        for agent in etat["agents"]:
            if agent["cle"] == cle:
                agent["statut"] = statut
                if detail:
                    agent["detail"] = detail
                if duree is not None:
                    agent["duree_s"] = round(duree, 2)
                if statut == "en_cours":
                    etat["agent_courant"] = agent["nom"]
                    etat["message"] = f"Agent {agent['numero']} — {agent['nom']} en cours…"
                break
        poids = etat.get("poids") or {a: 100 * p for a, p in POIDS_AGENTS.items()}
        acheves = {
            a["cle"] for a in etat["agents"]
            if a["statut"] in {"termine", "repli", "echec"}
        }
        termines = sum(poids.get(cle, 0) for cle in acheves)
        etat["pourcentage"] = (
            min(99, round(termines)) if etat["statut"] == "EN_COURS" else 100
        )

        # Temps restant : somme des durees predites des agents non acheves,
        # corrigee du rythme reellement observe sur ceux qui le sont deja.
        prevues = etat.get("durees_prevues") or {}
        if prevues:
            total_prevu = sum(prevues.values()) or 1.0
            restant = sum(v for cle, v in prevues.items() if cle not in acheves)
            mesure = sum(
                a["duree_s"] for a in etat["agents"]
                if a["cle"] in acheves and a.get("duree_s")
            )
            attendu = sum(prevues.get(cle, 0) for cle in acheves)

            # Correction du rythme, mais ponderee par la part de travail deja
            # accomplie. Extrapoler la duree totale a partir du seul Agent 1 —
            # le plus rapide et le moins representatif — donnerait une
            # estimation absurde des les premieres secondes. La confiance
            # accordee a la mesure croit donc avec l'avancement, et le modele
            # garde la main tant qu'on manque de recul.
            if attendu > 0.5 and mesure > 0:
                brut = min(4.0, max(0.25, mesure / attendu))
                confiance = min(1.0, attendu / total_prevu)
                facteur = 1.0 + (brut - 1.0) * confiance
            else:
                facteur = 1.0

            etat["temps_restant_s"] = round(restant * facteur, 1)
            etat["facteur_rythme"] = round(facteur, 2)


def _journaliser(analyse_id: str, niveau: str, message: str) -> None:
    with _LOCK:
        etat = PROGRESSION.get(analyse_id)
        if etat is not None:
            etat["journal"].append(
                {
                    "horodatage": datetime.datetime.now().strftime("%H:%M:%S"),
                    "niveau": niveau,
                    "message": message,
                }
            )


def obtenir_progression(analyse_id: str) -> dict | None:
    with _LOCK:
        etat = PROGRESSION.get(analyse_id)
        return dict(etat) if etat else None


def _nettoyer_pour_persistance(donnees):
    """
    Retire les objets non serialisables (matrices NumPy, instances de modeles)
    avant enregistrement en base : par convention, toute cle prefixee par « _ »
    est un artefact d'execution qui ne doit pas etre persiste.
    """
    if isinstance(donnees, dict):
        return {
            cle: _nettoyer_pour_persistance(valeur)
            for cle, valeur in donnees.items()
            if not str(cle).startswith("_")
        }
    if isinstance(donnees, list):
        return [_nettoyer_pour_persistance(valeur) for valeur in donnees]
    return donnees


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _executer_pipeline(analyse: dict, pdf_path: str) -> dict:
    """Enchaine les neuf agents. Ne leve jamais : les erreurs sont capturees."""
    analyse_id = analyse["id"]
    matiere = analyse["matiere"]
    niveau = analyse["niveau"]
    codes_pays = analyse.get("pays_reference") or None

    cle_referentiel = referentiels.cle_pour(matiere, niveau)
    notions = referentiels.notions_a_plat(cle_referentiel, codes_pays)
    analyse["referentiel_utilise"] = cle_referentiel
    analyse["nb_notions_referentiel"] = len(notions)

    # Socle de connaissances effectivement utilise. Sans cette trace, deux
    # analyses menees sur des versions differentes du referentiel seraient
    # comparees comme si elles partageaient la meme reference.
    signature = referentiels.signature_versions(cle_referentiel, codes_pays)
    analyse["referentiel_version"] = signature["signature"]
    analyse["referentiel_versions_par_pays"] = signature["par_pays"]
    analyse["referentiel_officiel"] = signature["entierement_officiel"]

    _journaliser(
        analyse_id,
        "info",
        f"Référentiel « {cle_referentiel} » chargé : {len(notions)} notions issues de "
        f"{len({n['code'] for n in notions})} pays "
        f"({'texte officiel' if signature['entierement_officiel'] else 'jeu reconstitué'}, "
        f"version {signature['signature']}).",
    )

    resultats: dict = {}

    def raffiner_prediction() -> None:
        """
        Reajuste les durees predites une fois les caracteristiques exactes
        connues. Avant l'Agent 1, la prediction repose sur la pre-extraction
        du module de depot ; apres l'Agent 3, elle dispose du nombre reel
        d'unites de contenu, qui est le facteur de cout dominant.
        """
        try:
            instantane = dict(analyse)
            instantane.update(resultats)
            prediction = prediction_execution.predire_durees(instantane)
        except Exception:
            return
        with _LOCK:
            etat = PROGRESSION.get(analyse_id)
            if etat:
                etat["poids"] = prediction["poids"]
                etat["durees_prevues"] = prediction["durees"]
                etat["duree_totale_prevue_s"] = prediction["duree_totale_prevue_s"]
                etat["prediction_source"] = prediction["source"]

    def executer(cle: str, libelle: str, fonction):
        """Execute un agent en tracant son etat et sa duree."""
        _maj_agent(analyse_id, cle, "en_cours")
        debut = time.time()
        try:
            resultat = fonction()
            duree = time.time() - debut
            _maj_agent(analyse_id, cle, "termine", duree=duree)
            _journaliser(analyse_id, "succes", f"{libelle} terminé en {duree:.1f} s.")
            return resultat
        except Exception as exc:
            duree = time.time() - debut
            _maj_agent(analyse_id, cle, "echec", detail=str(exc)[:200], duree=duree)
            _journaliser(analyse_id, "erreur", f"{libelle} en échec : {exc}")
            analyse.setdefault("incidents", []).append(
                {"agent": cle, "message": str(exc)[:300], "trace": traceback.format_exc()[-800:]}
            )
            if cle in AGENTS_CRITIQUES:
                raise
            return None

    # --- Agent 1 : Extraction ------------------------------------------
    resultats["agent1"] = executer(
        "agent1", "Agent 1 (Extraction)", lambda: agent1_extraction.process(pdf_path)
    )

    # --- Agent 2 : Comprehension ---------------------------------------
    resultats["agent2"] = executer(
        "agent2",
        "Agent 2 (Compréhension)",
        lambda: agent2_comprehension.process(resultats["agent1"], matiere, niveau),
    )
    if resultats["agent2"] is None:
        resultats["agent2"] = {"titre_cours": analyse["nom_fichier"], "chapitres": [],
                               "notions_cles_globales": [], "resume": "", "source": "indisponible"}
    elif resultats["agent2"].get("source") != "gemini":
        _maj_agent(analyse_id, "agent2", "repli",
                         detail="Modèle de langage indisponible — heuristique appliquée.")
        _journaliser(analyse_id, "alerte", "Agent 2 : repli déterministe activé.")

    # --- Agent 3 : Decoupage -------------------------------------------
    resultats["agent3"] = executer(
        "agent3",
        "Agent 3 (Découpage)",
        lambda: agent3_decoupage.process(resultats["agent1"], resultats["agent2"]),
    )
    _journaliser(
        analyse_id, "info",
        f"{resultats['agent3']['nb_unites']} unités de sens produites "
        f"(taille moyenne {resultats['agent3']['taille_moyenne_mots']} mots).",
    )
    # Les caracteristiques exactes du document sont maintenant connues :
    # la prediction de duree est reajustee pour la suite du pipeline.
    raffiner_prediction()

    # --- Agent 4 : Vectorisation ---------------------------------------
    resultats["agent4"] = executer(
        "agent4",
        "Agent 4 (Vectorisation)",
        lambda: agent4_vectorisation.process(resultats["agent3"], notions),
    )
    if resultats["agent4"].get("repli_actif"):
        _maj_agent(analyse_id, "agent4", "repli",
                         detail="Modèle neuronal indisponible — repli LSA (TF-IDF + SVD).")
        _journaliser(analyse_id, "alerte", "Agent 4 : repli statistique LSA activé.")
    else:
        _journaliser(
            analyse_id, "info",
            f"Vectorisation par {resultats['agent4']['moteur']} "
            f"({resultats['agent4']['dimension']} dimensions).",
        )

    # --- Agent 5 : Recherche -------------------------------------------
    resultats["agent5"] = executer(
        "agent5",
        "Agent 5 (Recherche)",
        lambda: agent5_recherche.process(resultats["agent3"], resultats["agent4"], notions),
    )
    _journaliser(
        analyse_id, "info",
        f"Base vectorielle : {resultats['agent5']['index']['moteur']} — "
        f"{resultats['agent5']['nb_notions_indexees']} vecteurs indexés.",
    )

    # --- Agent 6 : Comparaison -----------------------------------------
    resultats["agent6"] = executer(
        "agent6",
        "Agent 6 (Comparaison)",
        lambda: agent6_comparaison.process(
            resultats["agent2"], resultats["agent3"], resultats["agent4"],
            resultats["agent5"], matiere, niveau,
        ),
    )
    if not (resultats["agent6"].get("gemini") or {}).get("disponible"):
        _maj_agent(analyse_id, "agent6", "repli",
                         detail="Cartographie chiffrée produite ; analyse qualitative indisponible.")
        _journaliser(analyse_id, "alerte", "Agent 6 : analyse qualitative non générée.")

    # --- Agent 7 : Evaluation ------------------------------------------
    resultats["agent7"] = executer(
        "agent7",
        "Agent 7 (Évaluation)",
        lambda: agent7_evaluation.process(
            resultats["agent1"], resultats["agent2"], resultats["agent3"],
            resultats["agent4"], resultats["agent6"], matiere, niveau,
        ),
    )
    if resultats["agent7"] is None:
        resultats["agent7"] = {"note_globale": 0, "niveau_maturite": "Non evalue",
                               "indicateurs": {}, "contributions": [],
                               "appreciation": {"disponible": False}}

    # --- Agent 8 : Recommandations -------------------------------------
    resultats["agent8"] = executer(
        "agent8",
        "Agent 8 (Recommandations)",
        lambda: agent8_recommandations.process(
            resultats["agent2"], resultats["agent3"], resultats["agent4"],
            resultats["agent6"], resultats["agent7"], notions, matiere, niveau,
        ),
    )
    if resultats["agent8"] is None:
        resultats["agent8"] = {"recommandations": [], "plan_action": {}, "synthese": ""}
    else:
        parcours = resultats["agent8"].get("parcours_algorithmique") or {}
        if parcours.get("disponible"):
            _journaliser(
                analyse_id, "info",
                f"Parcours d'amélioration planifié : {parcours['nb_etapes']} étapes sur "
                f"{parcours['seances_planifiees']} séances — maîtrise pondérée "
                f"{parcours['etat_initial']['maitrise_ponderee']} → "
                f"{parcours['etat_final_predit']['maitrise_ponderee']}.",
            )
        if resultats["agent8"].get("source_redaction") == "repli_deterministe":
            _maj_agent(
                analyse_id, "agent8", "repli",
                detail="Parcours algorithmique conservé ; contenu pédagogique par gabarit.",
            )

    # --- Agent 9 : Rapport final ---------------------------------------
    resultats["agent9"] = executer(
        "agent9",
        "Agent 9 (Rapport final)",
        lambda: agent9_rapport.process(
            resultats["agent1"], resultats["agent2"], resultats["agent3"],
            resultats["agent4"], resultats["agent5"], resultats["agent6"],
            resultats["agent7"], resultats["agent8"], matiere, niveau,
        ),
    )
    if resultats["agent9"] is None:
        resultats["agent9"] = {"titre_rapport": analyse["nom_fichier"], "chiffres_cles": [],
                               "classement_referentiels": [], "synthese_executive": ""}
    elif not resultats["agent9"].get("synthese_generee_par_ia"):
        _maj_agent(analyse_id, "agent9", "repli",
                         detail="Synthèse exécutive produite sans modèle de langage.")

    analyse.update(resultats)
    return analyse


def _finaliser(analyse: dict, debut: float) -> None:
    analyse_id = analyse["id"]
    analyse["duree_analyse_s"] = round(time.time() - debut, 2)

    agent7 = analyse.get("agent7") or {}
    agent6 = analyse.get("agent6") or {}
    agent9 = analyse.get("agent9") or {}

    # Champs de synthese denormalises : ils alimentent l'historique et les
    # tableaux de bord sans avoir a recharger l'analyse complete.
    analyse["resume_note_globale"] = agent7.get("note_globale", 0)
    analyse["resume_maturite"] = agent7.get("niveau_maturite", "")
    analyse["resume_couverture_pct"] = agent6.get("score_global_pct", 0)
    analyse["resume_nb_manquantes"] = agent6.get("nb_notions_manquantes", 0)
    analyse["resume_nb_recommandations"] = len(
        (analyse.get("agent8") or {}).get("recommandations", [])
    )
    analyse["titre_cours"] = (analyse.get("agent2") or {}).get("titre_cours") or analyse["nom_fichier"]
    analyse["titre_rapport"] = agent9.get("titre_rapport", "")

    # --- Durees mesurees et diagnostic d'execution ----------------------
    # Elles sont persistees pour deux raisons : elles alimentent le modele de
    # prediction des analyses suivantes, et elles permettent de comparer le
    # profil de cette execution au profil normal appris.
    with _LOCK:
        etat = PROGRESSION.get(analyse_id) or {}
        # `is not None` et non un test de verite : un agent qui s'execute en
        # moins d'un centieme de seconde a bien une duree mesuree, et
        # l'ecarter fausserait le profil de repartition.
        durees = {
            a["cle"]: a["duree_s"] for a in etat.get("agents", [])
            if a.get("duree_s") is not None
        }
        prevues = dict(etat.get("durees_prevues") or {})
        source_prediction = etat.get("prediction_source")

    analyse["durees_agents"] = durees
    if durees:
        prevu_total = sum(prevues.values()) if prevues else None
        analyse["execution"] = {
            "durees_prevues": prevues,
            "duree_prevue_s": round(prevu_total, 1) if prevu_total else None,
            "duree_reelle_s": analyse["duree_analyse_s"],
            "ecart_prediction_pct": (
                round(100 * (analyse["duree_analyse_s"] - prevu_total) / prevu_total, 1)
                if prevu_total else None
            ),
            "source_prediction": source_prediction,
            "anomalie": prediction_execution.detecter_anomalie(durees),
        }
        anomalie = analyse["execution"]["anomalie"]
        if anomalie.get("atypique"):
            _journaliser(
                analyse_id, "alerte",
                f"Profil d'exécution atypique : {anomalie['agent_le_plus_ecarte']} occupe une "
                f"part inhabituelle du temps de traitement.",
            )

    database.enregistrer_analyse(_nettoyer_pour_persistance(analyse))
    # Les durees viennent de changer : le modele de prediction sera
    # reentraine au prochain demarrage pour en tenir compte.
    from app.services import entrainement

    entrainement.invalider("execution_duree")
    entrainement.invalider("execution_anomalie")

    with _LOCK:
        etat = PROGRESSION.get(analyse_id)
        if etat:
            etat["statut"] = analyse["statut"]
            etat["pourcentage"] = 100
            etat["duree_totale_s"] = analyse["duree_analyse_s"]
            etat["message"] = (
                "Analyse terminée." if analyse["statut"] == "TERMINEE" else "Analyse interrompue."
            )
            etat["erreur"] = analyse.get("erreur")


def _travail_de_fond(analyse: dict, pdf_path: str) -> None:
    debut = time.time()
    try:
        _executer_pipeline(analyse, pdf_path)
        analyse["statut"] = "TERMINEE"
        _journaliser(analyse["id"], "succes", "Rapport final disponible.")
    except Exception as exc:
        analyse["statut"] = "ECHEC"
        analyse["erreur"] = str(exc)
        _journaliser(analyse["id"], "erreur", f"Analyse interrompue : {exc}")
    finally:
        _finaliser(analyse, debut)


# ---------------------------------------------------------------------------
# Points d'entree publics
# ---------------------------------------------------------------------------

def creer_analyse(pdf_path: str, matiere: str, niveau: str, nom_fichier_original: str,
                  utilisateur: dict | None = None, pays_reference: list[str] | None = None) -> dict:
    """Cree l'enregistrement « Analyse » (equivalent de la classe Analyse du
    diagramme de classes) et le persiste au statut EN_COURS."""
    maintenant = datetime.datetime.now()
    analyse = {
        "id": uuid.uuid4().hex[:10],
        "date_creation": maintenant.strftime("%d/%m/%Y %H:%M"),
        "date_creation_iso": maintenant.isoformat(),
        "nom_fichier": nom_fichier_original,
        "chemin_fichier": pdf_path,
        "matiere": matiere,
        "niveau": niveau,
        "pays_reference": pays_reference or [],
        "statut": "EN_COURS",
        "utilisateur_id": (utilisateur or {}).get("id"),
        "utilisateur_email": (utilisateur or {}).get("email"),
        "utilisateur_nom": (utilisateur or {}).get("nom"),
        "erreur": None,
        "incidents": [],
    }
    database.creer_analyse(_nettoyer_pour_persistance(analyse))
    database.journaliser("analyse_lancee", analyse["utilisateur_id"],
                         {"analyse_id": analyse["id"], "fichier": nom_fichier_original})
    return analyse


def lancer_analyse_asynchrone(analyse: dict, pdf_path: str) -> str:
    """Demarre le pipeline dans un fil dedie et retourne l'identifiant."""
    _initialiser_progression(
        analyse["id"], analyse["nom_fichier"],
        prediction_execution.predire_durees(analyse),
    )
    fil = threading.Thread(
        target=_travail_de_fond, args=(analyse, pdf_path),
        name=f"analyse-{analyse['id']}", daemon=True,
    )
    fil.start()
    return analyse["id"]


def run_analysis(pdf_path: str, matiere: str, niveau: str, nom_fichier_original: str,
                 utilisateur: dict | None = None, pays_reference: list[str] | None = None) -> dict:
    """
    Execution **synchrone** du pipeline complet (utilisee par les tests et les
    scripts en ligne de commande). L'interface web passe par
    `creer_analyse()` + `lancer_analyse_asynchrone()`.
    """
    analyse = creer_analyse(pdf_path, matiere, niveau, nom_fichier_original,
                            utilisateur, pays_reference)
    _initialiser_progression(
        analyse["id"], analyse["nom_fichier"],
        prediction_execution.predire_durees(analyse),
    )
    _travail_de_fond(analyse, pdf_path)
    return analyse
