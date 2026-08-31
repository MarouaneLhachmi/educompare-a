"""
Routes principales — dépôt d'un support, suivi de l'analyse, consultation
et export du rapport.

Ces routes n'implementent aucune logique metier : elles se contentent
d'appeler les modules fonctionnels (Depot, Traitement et Analyse, Rapport et
Restitution), conformement a la separation des couches decrite au chapitre 3.
"""

import os

from flask import (
    Blueprint, abort, current_app, flash, jsonify, redirect,
    render_template, request, send_file, session, url_for,
)

from app.agents import CATALOGUE
from app.config import Config
from app.modules import CATALOGUE_MODULES
from app.modules import (
    module_auth_securite,
    module_depot_documents,
    module_historique_dashboard,
    module_rapport_restitution,
    module_traitement_analyse,
)
from app.services import database, prediction_execution, referentiels

bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Accueil et depot
# ---------------------------------------------------------------------------

@bp.route("/")
def accueil():
    utilisateur = module_auth_securite.utilisateur_courant()
    analyses_recentes = (
        database.lister_analyses(utilisateur_id=utilisateur["id"], limite=3)
        if utilisateur
        else []
    )
    return render_template(
        "accueil.html",
        matieres=referentiels.matieres(),
        niveaux=referentiels.niveaux(),
        catalogue_pays=referentiels.catalogue_pays(referentiels.cles_disponibles()[0]),
        statistiques_base=referentiels.statistiques(),
        agents=CATALOGUE,
        modules=CATALOGUE_MODULES,
        analyses_recentes=analyses_recentes,
    )


@bp.route("/api/referentiels")
def api_referentiels():
    """Renvoie les pays disponibles pour un couple matiere/niveau (selecteur dynamique)."""
    matiere = request.args.get("matiere", "")
    niveau = request.args.get("niveau", "")
    cle = referentiels.cle_pour(matiere, niveau)
    return jsonify({"cle": cle, "pays": referentiels.catalogue_pays(cle)})


@bp.route("/analyser", methods=["POST"])
def analyser():
    """
    Reception du document, validation de forme, triage documentaire, puis
    lancement asynchrone du pipeline.

    Quand le triage souleve une alerte — document hors perimetre, matiere
    incoherente, quasi-doublon — l'utilisateur est invite a confirmer plutot
    que d'etre bloque : les modeles de triage sont entraines sur un corpus
    encore reduit et n'ont pas vocation a arbitrer le travail d'un enseignant.
    """
    utilisateur = module_auth_securite.utilisateur_courant()

    matiere = request.form.get("matiere") or (referentiels.matieres() or ["Mathématiques"])[0]
    niveau = request.form.get("niveau") or (referentiels.niveaux() or [""])[0]
    pays_reference = request.form.getlist("pays")

    try:
        document = module_depot_documents.enregistrer_avec_triage(
            request.files.get("fichier_pdf"), utilisateur, matiere
        )
    except module_depot_documents.DocumentInvalide as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.accueil"))

    triage = document.get("triage") or {}
    alertes = triage.get("alertes") or []
    if alertes and request.form.get("confirme") != "1":
        # Le document est deja stocke : on met le depot en attente le temps
        # que l'utilisateur tranche, sans lui redemander le fichier.
        session["depot_en_attente"] = {
            "document": document,
            "matiere": matiere,
            "niveau": niveau,
            "pays": pays_reference,
        }
        # Duree prevue : le modele de prediction existe deja et sert la barre
        # de progression. L'exposer AVANT le lancement evite la question que
        # l'ecran de confirmation laissait sans reponse — « combien de temps
        # cela va-t-il prendre ? ».
        try:
            prediction = prediction_execution.predire_durees({
                "matiere": matiere,
                "niveau": niveau,
                "pays_reference": pays_reference,
                "document": document,
            })
        except Exception:
            prediction = None

        return render_template(
            "confirmation_depot.html",
            document=document,
            triage=triage,
            matiere=matiere,
            niveau=niveau,
            pays=pays_reference,
            prediction=prediction,
        )

    return _lancer(document, matiere, niveau, pays_reference, utilisateur)


@bp.route("/analyser/confirmer", methods=["POST"])
def analyser_confirmer():
    """Confirmation d'un depot signale par le triage."""
    attente = session.pop("depot_en_attente", None)
    if not attente:
        flash("La demande a expiré, merci de redéposer le document.", "error")
        return redirect(url_for("main.accueil"))

    utilisateur = module_auth_securite.utilisateur_courant()
    return _lancer(
        attente["document"], attente["matiere"], attente["niveau"],
        attente["pays"], utilisateur,
    )


def _lancer(document: dict, matiere: str, niveau: str, pays_reference: list[str],
            utilisateur: dict | None):
    """Cree l'analyse et demarre le pipeline."""
    analyse = module_traitement_analyse.creer_analyse(
        pdf_path=document["chemin"],
        matiere=matiere,
        niveau=niveau,
        nom_fichier_original=document["nom_original"],
        utilisateur=utilisateur,
        pays_reference=pays_reference,
    )
    analyse["document"] = document
    module_traitement_analyse.lancer_analyse_asynchrone(analyse, document["chemin"])
    return redirect(url_for("main.suivi", analyse_id=analyse["id"]))


# ---------------------------------------------------------------------------
# Suivi d'avancement
# ---------------------------------------------------------------------------

@bp.route("/analyse/<analyse_id>/suivi")
def suivi(analyse_id):
    """Ecran d'avancement temps reel du pipeline des neuf agents."""
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    if not module_auth_securite.peut_consulter_analyse(analyse):
        abort(403)
    if analyse.get("statut") == "TERMINEE" and not module_traitement_analyse.obtenir_progression(
        analyse_id
    ):
        return redirect(url_for("main.rapport", analyse_id=analyse_id))
    return render_template("suivi.html", analyse=analyse, agents=CATALOGUE)


@bp.route("/api/analyse/<analyse_id>/progression")
def api_progression(analyse_id):
    """Etat d'avancement consomme par la page de suivi (interrogation periodique)."""
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        return jsonify({"erreur": "Analyse introuvable."}), 404
    if not module_auth_securite.peut_consulter_analyse(analyse):
        return jsonify({"erreur": "Accès refusé."}), 403

    progression = module_traitement_analyse.obtenir_progression(analyse_id)
    if progression is None:
        # Le processus a redemarre : on reconstruit un etat terminal a partir
        # de ce qui a ete persiste en base.
        progression = {
            "analyse_id": analyse_id,
            "statut": analyse.get("statut", "INCONNU"),
            "pourcentage": 100 if analyse.get("statut") != "EN_COURS" else 0,
            "message": "État restauré depuis la base de données.",
            "agents": [],
            "journal": [],
            "erreur": analyse.get("erreur"),
        }

    progression.pop("debut", None)
    progression["url_rapport"] = url_for("main.rapport", analyse_id=analyse_id)
    return jsonify(progression)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------

@bp.route("/rapport/<analyse_id>")
def rapport(analyse_id):
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    if not module_auth_securite.peut_consulter_analyse(analyse):
        abort(403)
    if analyse.get("statut") == "EN_COURS":
        return redirect(url_for("main.suivi", analyse_id=analyse_id))
    if analyse.get("statut") == "ECHEC":
        return render_template("echec.html", analyse=analyse)

    contexte = module_rapport_restitution.build_report_context(analyse)
    return render_template("rapport.html", r=contexte)


@bp.route("/rapport/<analyse_id>/pdf")
def rapport_pdf(analyse_id):
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    if not module_auth_securite.peut_consulter_analyse(analyse):
        abort(403)
    if analyse.get("statut") != "TERMINEE":
        flash("Le rapport PDF n'est disponible qu'une fois l'analyse terminée.", "error")
        return redirect(url_for("main.suivi", analyse_id=analyse_id))

    chemin = os.path.join(Config.OUTPUT_FOLDER, f"rapport_{analyse_id}.pdf")
    try:
        module_rapport_restitution.export_pdf(analyse, chemin)
    except Exception as exc:
        current_app.logger.exception("Echec de generation du PDF : %s", exc)
        flash("La génération du PDF a échoué. Le rapport reste consultable en ligne.", "error")
        return redirect(url_for("main.rapport", analyse_id=analyse_id))

    nom = f"EduCompare_{analyse.get('matiere', 'rapport').replace(' ', '_')}_{analyse_id}.pdf"
    return send_file(chemin, as_attachment=True, download_name=nom)


@bp.route("/rapport/<analyse_id>/annexe")
def rapport_annexe(analyse_id):
    """
    Annexe d'accreditation (plan de transition, phase 2.3).

    Sortie distincte du rapport complet : releve notion par notion, preuve,
    provenance et angles morts declares, sans aucune phrase generee.
    """
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    if not module_auth_securite.peut_consulter_analyse(analyse):
        abort(403)
    if analyse.get("statut") != "TERMINEE":
        flash("L'annexe n'est disponible qu'une fois l'analyse terminée.", "error")
        return redirect(url_for("main.suivi", analyse_id=analyse_id))

    chemin = os.path.join(Config.OUTPUT_FOLDER, f"annexe_{analyse_id}.pdf")
    try:
        module_rapport_restitution.export_annexe_accreditation(analyse, chemin)
    except Exception as exc:
        current_app.logger.exception("Echec de generation de l'annexe : %s", exc)
        flash("La génération de l'annexe a échoué. Le rapport reste consultable en ligne.",
              "error")
        return redirect(url_for("main.rapport", analyse_id=analyse_id))

    nom = f"EduCompare_annexe_accreditation_{analyse_id}.pdf"
    return send_file(chemin, as_attachment=True, download_name=nom)


@bp.route("/rapport/<analyse_id>/retour", methods=["POST"])
def deposer_retour(analyse_id):
    """
    Boucle de retour enseignant (plan de transition, phase 1.2), **mode
    ombre** : l'etiquette est enregistree, mais rien dans le comportement du
    systeme n'en depend encore. La bascule effective (reentrainement de
    `couverture_clf`) n'aura lieu qu'une fois un volume suffisant accumule —
    voir `module_historique_dashboard`.

    Repond en JSON, appelee par une requete fetch() depuis le rapport ; ne
    redirige jamais pour ne pas perdre la position de lecture de la page.
    """
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        return jsonify({"erreur": "Analyse introuvable."}), 404
    if not module_auth_securite.peut_consulter_analyse(analyse):
        return jsonify({"erreur": "Accès refusé."}), 403

    donnees = request.get_json(silent=True) or {}
    type_retour = str(donnees.get("type", "")).strip()
    cle_notion = str(donnees.get("cle_notion", "")).strip()
    valeur = str(donnees.get("valeur", "")).strip()

    if type_retour not in database.TYPES_RETOUR or not cle_notion or not valeur:
        return jsonify({"erreur": "Retour incomplet."}), 400

    utilisateur = module_auth_securite.utilisateur_courant()
    database.enregistrer_retour(
        analyse_id, type_retour, cle_notion, valeur,
        utilisateur["id"] if utilisateur else None,
    )
    database.journaliser(
        "retour_enseignant", utilisateur["id"] if utilisateur else None,
        {"analyse_id": analyse_id, "type": type_retour},
    )
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Pages d'information
# ---------------------------------------------------------------------------

@bp.route("/architecture")
def architecture():
    """Presentation de l'architecture implementee (utile en soutenance)."""
    return render_template(
        "architecture.html",
        agents=CATALOGUE,
        modules=CATALOGUE_MODULES,
        sante=module_historique_dashboard.sante_composants(),
        base_connaissances=referentiels.statistiques(),
    )
