"""
Routes du back-office — reservees au role « administrateur ».

Elles couvrent les fonctionnalites etendues prevues pour cet acteur dans le
diagramme de cas d'utilisation : gestion des comptes utilisateurs et
consultation de l'ensemble des analyses realisees sur la plateforme, ainsi que
la supervision technique des composants (base de donnees, modele de langage,
base vectorielle).
"""

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for

from app.modules import (
    module_auth_securite, module_depot_documents, module_historique_dashboard,
    module_supervision_modeles,
)
from app.routes import _presentation
from app.services import anomalies_connexion, database

bp = Blueprint("admin", __name__, url_prefix="/administration")


@bp.route("/")
@module_auth_securite.administrateur_requis
def tableau_de_bord():
    donnees = module_historique_dashboard.tableau_de_bord_administrateur()
    return render_template("dashboard_admin.html", d=donnees)


@bp.route("/utilisateurs")
@module_auth_securite.administrateur_requis
def utilisateurs():
    recherche = request.args.get("q", "").strip().lower()
    comptes = database.lister_utilisateurs()
    if recherche:
        comptes = [
            u for u in comptes
            if recherche in str(u.get("email", "")).lower()
            or recherche in str(u.get("nom", "")).lower()
        ]
    # Nombre d'analyses par compte, affiche dans le tableau de gestion.
    analyses = database.lister_analyses()
    compteur = {}
    for analyse in analyses:
        compteur[analyse.get("utilisateur_id")] = compteur.get(analyse.get("utilisateur_id"), 0) + 1
    for compte in comptes:
        compte["nb_analyses"] = compteur.get(compte["id"], 0)

    return render_template("admin_utilisateurs.html", utilisateurs=comptes, recherche=recherche)


@bp.route("/analyses")
@module_auth_securite.administrateur_requis
def analyses():
    recherche = request.args.get("q", "").strip()
    matiere = request.args.get("matiere", "").strip()
    statut = request.args.get("statut", "").strip()

    toutes = database.lister_analyses()
    resultats = module_historique_dashboard.filtrer(toutes, recherche, matiere, statut)
    tri = request.args.get("tri", _presentation.TRI_PAR_DEFAUT)
    resultats = _presentation.trier_analyses(resultats, tri)
    pagination = _presentation.paginer(resultats, _presentation.page_demandee(request.args))

    return render_template(
        "historique.html",
        analyses=pagination["elements"],
        pagination=pagination,
        nb_filtres=len(resultats),
        total=len(toutes),
        recherche=recherche,
        matiere=matiere,
        statut=statut,
        tri=tri,
        tris=_presentation.TRIS_ANALYSES,
        matieres=sorted({a.get("matiere", "") for a in toutes if a.get("matiere")}),
        titre="Toutes les analyses de la plateforme",
        mode_admin=True,
    )


@bp.route("/systeme")
@module_auth_securite.administrateur_requis
def systeme():
    donnees = module_historique_dashboard.tableau_de_bord_administrateur()
    return render_template("admin_systeme.html", d=donnees)


@bp.route("/modeles")
@module_auth_securite.administrateur_requis
def modeles():
    """Module Supervision des Modèles : dérive des données et état des modèles."""
    return render_template(
        "admin_modeles.html",
        d=module_supervision_modeles.tableau_de_bord(),
        connexions=anomalies_connexion.connexions_atypiques(),
    )


@bp.route("/modeles/reentrainer", methods=["POST"])
@module_auth_securite.administrateur_requis
def reentrainer_modeles():
    """Invalide les modèles de modules pour forcer leur réentraînement."""
    nom = (request.form.get("modele") or "").strip()
    resultat = module_supervision_modeles.reentrainer([nom] if nom else None)
    database.journaliser(
        "reentrainement_modeles",
        module_auth_securite.utilisateur_courant()["id"],
        {"modeles": resultat["modeles_invalides"]},
    )
    flash(resultat["message"], "succes")
    return redirect(url_for("admin.modeles"))


# ---------------------------------------------------------------------------
# Actions de gestion des comptes
# ---------------------------------------------------------------------------

@bp.route("/utilisateurs/<utilisateur_id>/role", methods=["POST"])
@module_auth_securite.administrateur_requis
def changer_role(utilisateur_id):
    role = request.form.get("role", "utilisateur")
    courant = module_auth_securite.utilisateur_courant()

    if utilisateur_id == courant["id"] and role != "administrateur":
        flash("Vous ne pouvez pas retirer votre propre rôle administrateur.", "error")
        return redirect(url_for("admin.utilisateurs"))

    try:
        if not database.definir_role(utilisateur_id, role):
            abort(404)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("admin.utilisateurs"))

    flash(f"Rôle mis à jour : {role}.", "succes")
    return redirect(url_for("admin.utilisateurs"))


@bp.route("/utilisateurs/<utilisateur_id>/activation", methods=["POST"])
@module_auth_securite.administrateur_requis
def changer_activation(utilisateur_id):
    actif = request.form.get("actif") == "1"
    courant = module_auth_securite.utilisateur_courant()

    if utilisateur_id == courant["id"] and not actif:
        flash("Vous ne pouvez pas désactiver votre propre compte.", "error")
        return redirect(url_for("admin.utilisateurs"))

    if not database.definir_activation(utilisateur_id, actif):
        abort(404)
    flash("Compte activé." if actif else "Compte désactivé.", "succes")
    return redirect(url_for("admin.utilisateurs"))


@bp.route("/analyses/<analyse_id>/supprimer", methods=["POST"])
@module_auth_securite.administrateur_requis
def supprimer_analyse(analyse_id):
    analyse = database.analyse_par_id(analyse_id)
    if analyse is None:
        abort(404)
    module_depot_documents.supprimer_fichier(analyse.get("chemin_fichier", ""))
    database.supprimer_analyse(analyse_id)
    database.journaliser(
        "analyse_supprimee_admin",
        module_auth_securite.utilisateur_courant()["id"],
        {"analyse_id": analyse_id},
    )
    flash("Analyse supprimée de la plateforme.", "succes")
    return redirect(url_for("admin.analyses"))
