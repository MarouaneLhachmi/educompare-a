"""
Point d'entree de l'application EduCompare AI.

    python run.py

L'application est alors disponible sur http://127.0.0.1:5000
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").strip().lower() in {"1", "true", "yes"}
    # Le rechargement automatique reste desactive meme en mode debug : il
    # redemarre le processus et interromprait les analyses en cours d'execution
    # dans leur fil dedie (Module Traitement et Analyse).
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=debug,
        use_reloader=False,
        threaded=True,
    )
