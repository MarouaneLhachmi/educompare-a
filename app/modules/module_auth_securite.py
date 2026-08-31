"""
Module Authentification et Securite
====================================

Role (rapport de conception, section 3.3.1) : gerer l'ensemble du cycle de vie
de l'identite d'un utilisateur — creation de compte, connexion, gestion de
session, deconnexion — ainsi que la verification des habilitations pour chaque
action realisee sur le systeme. Il est sollicite en amont de toute autre
interaction avec la plateforme.

Choix technique (section 3.4.3) : l'authentification repose sur une connexion
a l'aide d'un **compte Google (OAuth 2.0 / OpenID Connect)**, ce qui evite au
systeme de stocker et de gerer des mots de passe. Les comptes ainsi crees sont
persistes dans MongoDB.

Deux roles sont definis (enumeration `Role` du diagramme de classes) :

- `utilisateur`      : depose des cours, consulte ses propres analyses ;
- `administrateur`   : acces au back-office (tous les utilisateurs, toutes les
                       analyses, supervision technique de la plateforme).
"""

from functools import wraps

from flask import session, redirect, url_for, flash, request, abort, g

from app.config import Config
from app.services import anomalies_connexion, database

CLE_SESSION = "utilisateur_id"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def ouvrir_session(utilisateur: dict) -> None:
    session.permanent = True
    session[CLE_SESSION] = utilisateur["id"]
    session["utilisateur_nom"] = utilisateur.get("nom")
    session["utilisateur_email"] = utilisateur.get("email")
    session["utilisateur_role"] = utilisateur.get("role", "utilisateur")
    database.journaliser("connexion", utilisateur["id"], {"email": utilisateur.get("email")})

    # Evaluation du comportement de connexion, apres journalisation pour que
    # la connexion courante fasse partie de l'historique analyse. Une
    # defaillance du detecteur ne doit jamais empecher quelqu'un d'entrer :
    # c'est un outil de supervision, pas un controle d'acces.
    try:
        diagnostic = anomalies_connexion.evaluer_connexion(utilisateur["id"])
        session["risque_connexion"] = diagnostic.get("risque")
        if diagnostic.get("atypique"):
            database.journaliser(
                "connexion_atypique", utilisateur["id"],
                {
                    "risque": diagnostic.get("risque"),
                    "methode": diagnostic.get("methode"),
                    "delai_s": (diagnostic.get("contexte") or {}).get("delai_s"),
                    "cadence_24h": (diagnostic.get("contexte") or {}).get("cadence_24h"),
                },
            )
    except Exception:
        pass


def fermer_session() -> None:
    utilisateur_id = session.get(CLE_SESSION)
    if utilisateur_id:
        database.journaliser("deconnexion", utilisateur_id, {})
    session.clear()


def utilisateur_courant() -> dict | None:
    """Retourne l'utilisateur connecte, avec mise en cache par requete."""
    if "utilisateur" in g:
        return g.utilisateur
    utilisateur_id = session.get(CLE_SESSION)
    utilisateur = database.utilisateur_par_id(utilisateur_id) if utilisateur_id else None
    if utilisateur and not utilisateur.get("actif", True):
        # Compte desactive par un administrateur pendant la session.
        session.clear()
        utilisateur = None
    g.utilisateur = utilisateur
    return utilisateur


def est_connecte() -> bool:
    return utilisateur_courant() is not None


def est_administrateur() -> bool:
    utilisateur = utilisateur_courant()
    return bool(utilisateur and utilisateur.get("role") == "administrateur")


# ---------------------------------------------------------------------------
# Habilitations
# ---------------------------------------------------------------------------

def connexion_requise(vue):
    """Interdit l'acces a une vue aux visiteurs non authentifies."""

    @wraps(vue)
    def enveloppe(*args, **kwargs):
        if not est_connecte():
            flash("Connectez-vous pour accéder à cet espace.", "info")
            return redirect(url_for("auth.connexion", suivant=request.path))
        return vue(*args, **kwargs)

    return enveloppe


def administrateur_requis(vue):
    """Reserve une vue aux comptes disposant du role administrateur."""

    @wraps(vue)
    def enveloppe(*args, **kwargs):
        if not est_connecte():
            flash("Connectez-vous pour accéder à cet espace.", "info")
            return redirect(url_for("auth.connexion", suivant=request.path))
        if not est_administrateur():
            abort(403)
        return vue(*args, **kwargs)

    return enveloppe


def peut_consulter_analyse(analyse: dict) -> bool:
    """
    Regle d'habilitation sur une analyse : son proprietaire ou un
    administrateur. Les analyses anonymes (mode demonstration sans compte)
    restent consultables par le porteur du lien.
    """
    if analyse is None:
        return False
    if est_administrateur():
        return True
    utilisateur = utilisateur_courant()
    proprietaire = analyse.get("utilisateur_id")
    if proprietaire is None:
        return True
    return bool(utilisateur and utilisateur["id"] == proprietaire)


def peut_consulter_programme(programme: dict) -> bool:
    """
    Regle d'habilitation sur un programme, calquee sur celle des analyses :
    son proprietaire ou un administrateur.

    Le rattachement d'une analyse est verifie separement, contre
    `peut_consulter_analyse` : un programme ne doit pas devenir un moyen de
    lire l'analyse d'autrui.
    """
    if programme is None:
        return False
    if est_administrateur():
        return True
    utilisateur = utilisateur_courant()
    proprietaire = programme.get("utilisateur_id")
    if proprietaire is None:
        return True
    return bool(utilisateur and utilisateur["id"] == proprietaire)


# ---------------------------------------------------------------------------
# Creation / connexion des comptes
# ---------------------------------------------------------------------------

def connecter_via_google(profil_google: dict) -> dict:
    """
    Cree ou met a jour le compte a partir du profil OpenID renvoye par Google,
    puis ouvre la session.
    """
    utilisateur = database.enregistrer_ou_mettre_a_jour_utilisateur(
        {
            "email": profil_google.get("email"),
            "nom": profil_google.get("name") or profil_google.get("given_name"),
            "photo": profil_google.get("picture"),
            "fournisseur": "google",
        }
    )
    ouvrir_session(utilisateur)
    return utilisateur


def peut_obtenir_role_admin(email: str) -> bool:
    """
    Regle d'attribution du role administrateur en mode demonstration.

    Le role n'est jamais accorde sur simple demande : il l'est uniquement si
    l'adresse figure dans ADMIN_EMAILS, ou s'il s'agit du **tout premier
    compte** de l'instance (amorcage — il faut bien un premier administrateur
    pour pouvoir promouvoir les suivants depuis le back-office).
    """
    if (email or "").strip().lower() in Config.ADMIN_EMAILS:
        return True
    return not any(
        u.get("role") == "administrateur" for u in database.lister_utilisateurs()
    )


def connecter_en_demonstration(email: str, nom: str | None = None,
                               administrateur: bool = False) -> dict:
    """
    Connexion de secours **sans Google**, destinee a la soutenance lorsque
    aucune application OAuth n'a ete declaree sur la console Google Cloud.
    Elle est desactivable par la variable d'environnement ALLOW_DEMO_LOGIN.

    Le role administrateur n'est pas accorde librement : voir
    `peut_obtenir_role_admin()`.
    """
    if not Config.ALLOW_DEMO_LOGIN:
        raise PermissionError("La connexion de démonstration est désactivée sur cette instance.")

    email = (email or "").strip().lower()
    promotion_autorisee = administrateur and peut_obtenir_role_admin(email)

    utilisateur = database.enregistrer_ou_mettre_a_jour_utilisateur(
        {
            "email": email,
            "nom": nom or email.split("@")[0].replace(".", " ").title(),
            "photo": None,
            "fournisseur": "demonstration",
        }
    )
    if promotion_autorisee and utilisateur.get("role") != "administrateur":
        database.definir_role(utilisateur["id"], "administrateur")
        utilisateur = database.utilisateur_par_id(utilisateur["id"])

    ouvrir_session(utilisateur)
    utilisateur = dict(utilisateur)
    utilisateur["promotion_refusee"] = bool(administrateur and not promotion_autorisee)
    return utilisateur


def contexte_modele() -> dict:
    """Variables injectees dans tous les templates Jinja."""
    utilisateur = utilisateur_courant()
    return {
        "utilisateur_courant": utilisateur,
        "est_administrateur": bool(utilisateur and utilisateur.get("role") == "administrateur"),
        "google_actif": Config.google_oauth_configured(),
        "demo_login_actif": Config.ALLOW_DEMO_LOGIN,
    }
