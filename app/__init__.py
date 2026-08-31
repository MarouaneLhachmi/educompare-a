"""
Fabrique de l'application Flask — EduCompare AI.

L'application implemente les quatre couches decrites au chapitre 3 du rapport
de conception :

- **Couche de presentation** : templates Jinja2 + feuille de style et scripts
  de `app/static` (interface de depot, tableaux de bord, rapport).
- **Couche logique metier** : blueprints de `app/routes` + les cinq modules
  fonctionnels de `app/modules` (authentification, depot, orchestration,
  restitution, historique).
- **Couche intelligence et donnees** : les neuf agents de `app/agents`,
  la base vectorielle FAISS et le client du modele de langage.
- **Couche base de donnees** : MongoDB, encapsulee dans `app/services/database`.

Le prototype d'origine (Flask + 2 agents) est ici etendu a la totalite de
l'architecture cible, en Python, afin de disposer d'une implementation
executable de bout en bout.
"""

import os

from dotenv import load_dotenv
from flask import Flask, render_template
from markupsafe import escape, Markup

load_dotenv()  # charge les variables du fichier .env s'il existe

oauth = None  # instancie dans create_app() si les identifiants Google existent


def _enregistrer_oauth(app: Flask):
    """Declare le fournisseur d'identite Google (OpenID Connect) via Authlib."""
    global oauth
    from app.config import Config

    if not Config.google_oauth_configured():
        return None
    try:
        from authlib.integrations.flask_client import OAuth
    except ImportError:
        app.logger.warning("Authlib n'est pas installe : connexion Google desactivee.")
        return None

    oauth = OAuth(app)
    oauth.register(
        name="google",
        client_id=Config.GOOGLE_CLIENT_ID,
        client_secret=Config.GOOGLE_CLIENT_SECRET,
        server_metadata_url=Config.GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile"},
    )
    return oauth


def _enregistrer_filtres(app: Flask) -> None:
    """Filtres Jinja utilises par les templates."""

    @app.template_filter("pourcentage")
    def pourcentage(valeur, decimales=0):
        try:
            return f"{float(valeur):.{decimales}f} %"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("duree")
    def duree(secondes):
        try:
            secondes = float(secondes)
        except (TypeError, ValueError):
            return "—"
        if secondes < 60:
            return f"{secondes:.1f} s"
        return f"{int(secondes // 60)} min {int(secondes % 60)} s"

    @app.template_filter("classe_statut")
    def classe_statut(statut):
        return {
            "Couverte": "ok",
            "Partiellement couverte": "moyen",
            "Non couverte": "ko",
            "TERMINEE": "ok",
            "EN_COURS": "encours",
            "ECHEC": "ko",
        }.get(statut, "neutre")

    @app.template_filter("initiales")
    def initiales(nom):
        morceaux = [m for m in str(nom or "?").replace(".", " ").split() if m]
        return "".join(m[0].upper() for m in morceaux[:2]) or "?"

    @app.template_filter("surligner")
    def surligner(texte, terme):
        """Met en evidence un terme de recherche dans un extrait (sans XSS)."""
        texte = escape(str(texte or ""))
        terme = str(terme or "").strip()
        if not terme:
            return texte
        minuscule = str(texte).lower()
        position = minuscule.find(terme.lower())
        if position == -1:
            return texte
        brut = str(texte)
        fin = position + len(terme)
        return Markup(f"{brut[:position]}<mark>{brut[position:fin]}</mark>{brut[fin:]}")


def _enregistrer_composants(app: Flask) -> None:
    """
    Rend les macros de `_composants.html` disponibles dans tous les gabarits
    sous le nom `c`, sans import local.

    Un `{% import %}` place dans `base.html` ne serait pas visible depuis les
    gabarits qui l'etendent : en Jinja, un bloc enfant ne voit pas les imports
    du parent. Les enregistrer comme variable globale de l'environnement est
    la seule facon d'obtenir « defini une fois, disponible partout ».

    Aucune macro n'appelle `url_for` : le module est donc charge hors contexte
    de requete, une seule fois au demarrage.
    """
    app.jinja_env.globals["c"] = app.jinja_env.get_template("_composants.html").module


def _prechauffer(app: Flask) -> None:
    """
    Charge l'encodeur semantique dans un fil de fond, des le demarrage.

    Le premier chargement du modele `sentence-transformers` coute une
    vingtaine de secondes. Sans prechauffage, ce cout serait paye par le
    premier utilisateur qui depose un document — au moment precis ou le
    module de depot lui promet un controle instantane. Le faire en arriere
    plan au demarrage le rend invisible.

    L'echec du prechauffage est sans consequence : chaque appelant sait
    fonctionner sans l'encodeur.
    """
    import threading

    def charger():
        try:
            from app.agents import agent4_vectorisation

            agent4_vectorisation.charger_modele_transformer()
            app.logger.info("Encodeur semantique prechauffe.")
        except Exception as exc:  # pragma: no cover
            app.logger.warning("Prechauffage de l'encodeur impossible : %s", exc)

    threading.Thread(target=charger, name="prechauffage-encodeur", daemon=True).start()


def create_app() -> Flask:
    from app.config import Config

    app = Flask(__name__)
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 12  # 12 h
    # Jinja compile et met en cache les templates au premier rendu. Comme le
    # rechargement automatique du code est desactive (il interromprait les
    # analyses en cours), une modification de gabarit resterait invisible
    # jusqu'au redemarrage. Le cout d'un `stat()` par rendu est negligeable
    # au regard du temps d'une analyse.
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

    _enregistrer_oauth(app)
    _enregistrer_filtres(app)
    _enregistrer_composants(app)

    from app.routes.main import bp as bp_main
    from app.routes.auth import bp as bp_auth
    from app.routes.dashboard import bp as bp_dashboard
    from app.routes.programmes import bp as bp_programmes
    from app.routes.admin import bp as bp_admin

    app.register_blueprint(bp_main)
    app.register_blueprint(bp_auth)
    app.register_blueprint(bp_dashboard)
    app.register_blueprint(bp_programmes)
    app.register_blueprint(bp_admin)

    from app.modules import module_auth_securite

    _prechauffer(app)

    @app.context_processor
    def injecter_contexte():
        contexte = module_auth_securite.contexte_modele()
        contexte["nom_application"] = "EduCompare AI"
        return contexte

    # ------------------------------------------------------------------
    # Pages d'erreur
    # ------------------------------------------------------------------
    @app.errorhandler(403)
    def acces_refuse(_):
        return render_template(
            "erreur.html",
            code=403,
            titre="Accès refusé",
            message="Cet espace est réservé aux administrateurs de la plateforme.",
        ), 403

    @app.errorhandler(404)
    def introuvable(_):
        return render_template(
            "erreur.html",
            code=404,
            titre="Page introuvable",
            message="La ressource demandée n'existe pas ou a été supprimée.",
        ), 404

    @app.errorhandler(413)
    def trop_volumineux(_):
        limite = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
        return render_template(
            "erreur.html",
            code=413,
            titre="Fichier trop volumineux",
            message=f"Le document dépasse la taille maximale autorisée ({limite} Mo).",
        ), 413

    @app.errorhandler(500)
    def erreur_serveur(exc):
        app.logger.exception("Erreur interne : %s", exc)
        return render_template(
            "erreur.html",
            code=500,
            titre="Erreur interne",
            message="Une erreur inattendue est survenue pendant le traitement.",
        ), 500

    return app
