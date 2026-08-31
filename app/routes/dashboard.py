"""
Routes de l'espace utilisateur — Module Historique et Tableau de Bord.

Perimetre strictement limite aux analyses de l'utilisateur connecte.
"""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.modules import module_auth_securite, module_depot_documents, module_historique_dashboard
from app.routes import _presentation
from app.services import database

bp = Blueprint("dashboard", __name__, url_prefix="/espace")


@bp.route("/")
@module_auth_securite.connexion_requise
def tableau_de_bord():
    utilisateur = module_auth_securite.utilisateur_courant()
    donnees = module_historique_dashboard.tableau_de_bord_utilisateur(utilisateur)
    return render_template("dashboard_utilisateur.html", d=donnees, utilisateur=utilisateur)


@bp.route("/historique")
@module_auth_securite.connexion_requise
def historique():
    utilisateur = module_auth_securite.utilisateur_courant()
    recherche = request.args.get("q", "").strip()
    matiere = request.args.get("matiere", "").strip()
    statut = request.args.get("statut", "").strip()

    analyses = database.lister_analyses(utilisateur_id=utilisateur["id"])
    resultats = module_historique_dashboard.filtrer(analyses, recherche, matiere, statut)
    tri = request.args.get("tri", _presentation.TRI_PAR_DEFAUT)
    resultats = _presentation.trier_analyses(resultats, tri)
    pagination = _presentation.paginer(resultats, _presentation.page_demandee(request.args))

    return render_template(
        "historique.html",
        analyses=pagination["elements"],
        pagination=pagination,
        nb_filtres=len(resultats),
        total=len(analyses),
        recherche=recherche,
        matiere=matiere,
        statut=statut,
        tri=tri,
        tris=_presentation.TRIS_ANALYSES,
        matieres=sorted({a.get("matiere", "") for a in analyses if a.get("matiere")}),
        titre="Mon historique d'analyses",
        mode_admin=False,
    )


@bp.route("/profil")
@module_auth_securite.connexion_requise
def profil():
    utilisateur = module_auth_securite.utilisateur_courant()
    donnees = module_historique_dashboard.tableau_de_bord_utilisateur(utilisateur)
    return render_template("profil.html", utilisateur=utilisateur, d=donnees)


@bp.route("/analyse/<analyse_id>/supprimer", methods=["POST"])
@module_auth_securite.connexion_requise
def supprimer(analyse_id):
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    if not module_auth_securite.peut_consulter_analyse(analyse):
        abort(403)

    module_depot_documents.supprimer_fichier(analyse.get("chemin_fichier", ""))
    database.supprimer_analyse(analyse_id)
    database.journaliser(
        "analyse_supprimee",
        module_auth_securite.utilisateur_courant()["id"],
        {"analyse_id": analyse_id},
    )
    flash("Analyse supprimée.", "succes")
    return redirect(url_for("dashboard.historique"))
