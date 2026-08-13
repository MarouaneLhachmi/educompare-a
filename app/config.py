"""
Configuration centrale de l'application EduCompare AI.

Toutes les valeurs sont lues depuis les variables d'environnement (fichier
`.env`), avec des valeurs par defaut raisonnables pour que l'application
reste demarrable meme sans configuration complete.
"""

import os


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off"}


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # ------------------------------------------------------------------
    # Flask
    # ------------------------------------------------------------------
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "educompare-ai-demo-secret")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024

    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.environ.get("UPLOAD_FOLDER", "uploads"))
    OUTPUT_FOLDER = os.path.join(BASE_DIR, os.environ.get("OUTPUT_FOLDER", "outputs"))

    ALLOWED_EXTENSIONS = {"pdf"}

    # ------------------------------------------------------------------
    # Couche intelligence (Agents)
    # ------------------------------------------------------------------
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "90"))

    USE_EMBEDDINGS = _env_bool("USE_EMBEDDINGS", True)
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Re-ranking par cross-encodeur (Agent 6) ----------------------
    # Le bi-encodeur mesure une proximite thematique ; le cross-encodeur
    # evalue conjointement la paire (unite de cours, notion de referentiel)
    # et corrige l'essentiel des faux positifs. Desactivable si la machine
    # de demonstration ne peut pas heberger le modele.
    USE_CROSS_ENCODER = _env_bool("USE_CROSS_ENCODER", True)
    CROSS_ENCODER_MODEL = os.environ.get(
        "CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    # Nombre de notions candidates re-scorees par unite de cours.
    CROSS_ENCODER_TOP_K = int(os.environ.get("CROSS_ENCODER_TOP_K", "3"))

    # --- Modeles entraines hors ligne (notebooks Colab) ---------------
    # Artefacts deposes dans ce dossier ; chaque agent dispose d'un repli
    # complet si l'artefact correspondant est absent.
    MODELS_FOLDER = os.path.join(BASE_DIR, os.environ.get("MODELS_FOLDER", "app/models"))

    # --- Planification du parcours (Agent 8) --------------------------
    # Nombre maximal d'etapes du parcours d'amelioration genere.
    PARCOURS_MAX_ETAPES = int(os.environ.get("PARCOURS_MAX_ETAPES", "12"))
    # Etapes pour lesquelles le contenu pedagogique complet (theorie +
    # exercices) est redige par le modele de langage.
    PARCOURS_ETAPES_REDIGEES = int(os.environ.get("PARCOURS_ETAPES_REDIGEES", "5"))

    # ------------------------------------------------------------------
    # Couche base de donnees (MongoDB)
    # ------------------------------------------------------------------
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "educompare_ai")
    MONGO_TIMEOUT_MS = int(os.environ.get("MONGO_TIMEOUT_MS", "2000"))

    # ------------------------------------------------------------------
    # Authentification (OAuth Google)
    # ------------------------------------------------------------------
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    GOOGLE_DISCOVERY_URL = (
        "https://accounts.google.com/.well-known/openid-configuration"
    )
    # Comptes promus automatiquement au role "administrateur" a la connexion.
    ADMIN_EMAILS = _env_list("ADMIN_EMAILS")
    # Connexion de demonstration (sans Google) : utile pour la soutenance si
    # aucune application OAuth n'a ete declaree sur la console Google Cloud.
    ALLOW_DEMO_LOGIN = _env_bool("ALLOW_DEMO_LOGIN", True)

    @staticmethod
    def google_oauth_configured() -> bool:
        return bool(Config.GOOGLE_CLIENT_ID and Config.GOOGLE_CLIENT_SECRET)

    @staticmethod
    def gemini_configured() -> bool:
        key = Config.GEMINI_API_KEY
        return bool(key) and key != "colle_ta_cle_ici"
