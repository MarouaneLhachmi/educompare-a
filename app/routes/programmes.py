"""
Routes des programmes — du cours au cursus (phase 2.1 du plan de transition).

Un programme regroupe les analyses des documents d'un meme cursus. Ces routes
ne calculent rien : elles delegent le tri et l'agregation au module Historique
et Tableau de Bord, qui delegue lui-meme la decision de couverture a l'Agent 6
en mode agrege.
"""

from flask import (
    Blueprint, abort, flash, redirect, render_template, request, url_for,
)

from app.modules import module_auth_securite, module_historique_dashboard
from app.services import database, referentiels

bp = Blueprint("programmes", __name__, url_prefix="/programmes")


@bp.route("/")
@module_auth_securite.connexion_requise
def liste():
    utilisateur = module_auth_securite.utilisateur_courant()
    if module_auth_securite.est_administrateur():
        mes_programmes = database.lister_programmes()
    else:
        mes_programmes = database.lister_programmes(utilisateur_id=utilisateur["id"])

    # Compte des documents rattaches, pour la liste : le detail de la
    # couverture n'est calcule qu'a l'ouverture d'un programme.
    for programme in mes_programmes:
        programme["nb_documents"] = len(programme.get("analyse_ids") or [])

    return render_template(
        "programmes.html",
        programmes=mes_programmes,
        matieres=referentiels.matieres(),
        niveaux=referentiels.niveaux(),
        utilisateur=utilisateur,
    )


@bp.route("/creer", methods=["POST"])
@module_auth_securite.connexion_requise
def creer():
    utilisateur = module_auth_securite.utilisateur_courant()
    nom = (request.form.get("nom") or "").strip()
    if not nom:
        flash("Le programme doit avoir un nom.", "error")
        return redirect(url_for("programmes.liste"))

    programme = database.creer_programme({
        "nom": nom,
        "matiere": request.form.get("matiere") or "",
        "niveau": request.form.get("niveau") or "",
        "annee": (request.form.get("annee") or "").strip(),
        "etablissement": (request.form.get("etablissement") or "").strip(),
        "utilisateur_id": utilisateur["id"],
        "utilisateur_nom": utilisateur.get("nom"),
    })
    database.journaliser("programme_cree", utilisateur["id"],
                         {"programme_id": programme["id"], "nom": nom})
    flash(f"Programme « {nom} » créé.", "succes")
    return redirect(url_for("programmes.detail", programme_id=programme["id"]))


@bp.route("/<programme_id>")
@module_auth_securite.connexion_requise
def detail(programme_id):
    programme = database.programme_par_id(programme_id)
    if programme is None:
        abort(404)
    if not module_auth_securite.peut_consulter_programme(programme):
        abort(403)

    utilisateur = module_auth_securite.utilisateur_courant()
    vue = module_historique_dashboard.vue_programme(programme)
    vue["rattachables"] = module_historique_dashboard.analyses_rattachables(
        programme, utilisateur
    )
    return render_template("programme.html", v=vue, utilisateur=utilisateur)


@bp.route("/<programme_id>/rattacher", methods=["POST"])
@module_auth_securite.connexion_requise
def rattacher(programme_id):
    programme = database.programme_par_id(programme_id)
    if programme is None:
        abort(404)
    if not module_auth_securite.peut_consulter_programme(programme):
        abort(403)

    analyse_id = (request.form.get("analyse_id") or "").strip()
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        flash("Analyse introuvable.", "error")
    elif not module_auth_securite.peut_consulter_analyse(analyse):
        # On ne rattache pas une analyse qu'on n'a pas le droit de lire :
        # sinon le programme deviendrait un contournement du cloisonnement.
        abort(403)
    elif database.rattacher_analyse(programme_id, analyse_id):
        flash(f"« {analyse.get('nom_fichier')} » rattaché au programme.", "succes")
    else:
        flash("Ce document est déjà rattaché à ce programme.", "info")

    return redirect(url_for("programmes.detail", programme_id=programme_id))


@bp.route("/<programme_id>/detacher/<analyse_id>", methods=["POST"])
@module_auth_securite.connexion_requise
def detacher(programme_id, analyse_id):
    programme = database.programme_par_id(programme_id)
    if programme is None:
        abort(404)
    if not module_auth_securite.peut_consulter_programme(programme):
        abort(403)

    if database.detacher_analyse(programme_id, analyse_id):
        flash("Document retiré du programme. L'analyse elle-même est conservée.",
              "succes")
    return redirect(url_for("programmes.detail", programme_id=programme_id))


@bp.route("/<programme_id>/supprimer", methods=["POST"])
@module_auth_securite.connexion_requise
def supprimer(programme_id):
    programme = database.programme_par_id(programme_id)
    if programme is None:
        abort(404)
    if not module_auth_securite.peut_consulter_programme(programme):
        abort(403)

    database.supprimer_programme(programme_id)
    database.journaliser("programme_supprime",
                         (module_auth_securite.utilisateur_courant() or {}).get("id"),
                         {"programme_id": programme_id})
    flash("Programme supprimé. Les analyses rattachées sont conservées.", "succes")
    return redirect(url_for("programmes.liste"))
