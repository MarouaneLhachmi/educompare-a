"""
Routes d'authentification — Module Authentification et Securite.

Parcours nominal : connexion via un compte Google (OAuth 2.0 / OpenID
Connect). Un parcours de secours (« connexion de demonstration ») permet de
presenter la plateforme sans dependre de la console Google Cloud ; il est
desactivable par configuration.
"""

from urllib.parse import urlparse

from flask import (
    Blueprint, current_app, flash, redirect, render_template, request, session, url_for,
)

from app.config import Config
from app.modules import module_auth_securite

bp = Blueprint("auth", __name__)


def _redirection_sure(cible: str | None, defaut: str) -> str:
    """N'autorise que les redirections internes (protection open redirect)."""
    if not cible:
        return defaut
    analyse = urlparse(cible)
    if analyse.scheme or analyse.netloc or not cible.startswith("/"):
        return defaut
    return cible


@bp.route("/connexion")
def connexion():
    if module_auth_securite.est_connecte():
        return redirect(url_for("dashboard.tableau_de_bord"))
    return render_template(
        "connexion.html",
        suivant=_redirection_sure(request.args.get("suivant"), url_for("dashboard.tableau_de_bord")),
    )


@bp.route("/connexion/google")
def connexion_google():
    """Redirige vers l'ecran de consentement Google."""
    import app as application

    if application.oauth is None:
        flash(
            "La connexion Google n'est pas configurée sur cette instance "
            "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET absents du fichier .env).",
            "error",
        )
        return redirect(url_for("auth.connexion"))

    session["suivant"] = _redirection_sure(
        request.args.get("suivant"), url_for("dashboard.tableau_de_bord")
    )
    redirection = url_for("auth.callback_google", _external=True)
    return application.oauth.google.authorize_redirect(redirection)


@bp.route("/connexion/google/callback")
def callback_google():
    """Point de retour OAuth : recupere le profil et ouvre la session."""
    import app as application

    if application.oauth is None:
        return redirect(url_for("auth.connexion"))

    try:
        jeton = application.oauth.google.authorize_access_token()
        profil = jeton.get("userinfo") or application.oauth.google.userinfo(token=jeton)
    except Exception as exc:
        current_app.logger.warning("Echec OAuth Google : %s", exc)
        flash("La connexion Google a échoué. Merci de réessayer.", "error")
        return redirect(url_for("auth.connexion"))

    if not profil or not profil.get("email"):
        flash("Google n'a pas transmis d'adresse e-mail exploitable.", "error")
        return redirect(url_for("auth.connexion"))

    utilisateur = module_auth_securite.connecter_via_google(dict(profil))
    flash(f"Bienvenue {utilisateur.get('nom')} !", "succes")
    return redirect(session.pop("suivant", None) or url_for("dashboard.tableau_de_bord"))


@bp.route("/connexion/demonstration", methods=["POST"])
def connexion_demonstration():
    """Connexion de secours par simple adresse e-mail (mode demonstration)."""
    if not Config.ALLOW_DEMO_LOGIN:
        flash("La connexion de démonstration est désactivée sur cette instance.", "error")
        return redirect(url_for("auth.connexion"))

    email = (request.form.get("email") or "").strip().lower()
    nom = (request.form.get("nom") or "").strip() or None
    demander_admin = request.form.get("role") == "administrateur"

    if "@" not in email or "." not in email.split("@")[-1]:
        flash("Merci de saisir une adresse e-mail valide.", "error")
        return redirect(url_for("auth.connexion"))

    try:
        utilisateur = module_auth_securite.connecter_en_demonstration(
            email, nom, administrateur=demander_admin
        )
    except PermissionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("auth.connexion"))

    if utilisateur.get("promotion_refusee"):
        flash(
            "Le rôle administrateur n'a pas été accordé : cette adresse ne figure pas "
            "dans ADMIN_EMAILS et un administrateur existe déjà. Demandez à un "
            "administrateur de vous promouvoir depuis le back-office.",
            "alerte",
        )
    flash(f"Connecté en mode démonstration ({utilisateur['role']}).", "succes")
    suivant = _redirection_sure(
        request.form.get("suivant"), url_for("dashboard.tableau_de_bord")
    )
    return redirect(suivant)


@bp.route("/deconnexion")
def deconnexion():
    module_auth_securite.fermer_session()
    flash("Vous avez été déconnecté.", "info")
    return redirect(url_for("main.accueil"))
