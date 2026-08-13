"""
A executer UNE SEULE FOIS, avec une connexion Internet, pour telecharger et
mettre en cache localement le modele semantique multilingue utilise par
l'Agent 4 (Vectorisation) — et donc, indirectement, par les Agents 5 et 6.

Une fois telecharge, le modele est relu depuis le cache local : les analyses
fonctionnent alors entierement hors-ligne, sans appel reseau.

Usage :
    python setup_download_model.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


def main() -> int:
    print(f"Telechargement du modele semantique « {MODEL_NAME} »...")
    print("(quelques minutes selon la connexion — environ 470 Mo)")
    try:
        from sentence_transformers import SentenceTransformer

        from app.agents.agent4_vectorisation import _dimension_du_modele

        modele = SentenceTransformer(MODEL_NAME)
        dimension = _dimension_du_modele(modele)
        # Verification fonctionnelle : le modele doit rapprocher une phrase
        # francaise de sa traduction anglaise (propriete cross-lingue).
        import numpy as np

        vecteurs = modele.encode(
            ["fractions et nombres decimaux", "fractions and decimal numbers"],
            normalize_embeddings=True,
        )
        similarite = float(np.dot(vecteurs[0], vecteurs[1]))

        print(f"OK : modele telecharge et mis en cache ({dimension} dimensions).")
        print(f"Verification cross-lingue FR/EN : similarite = {similarite:.3f}")
        print("Les Agents 4, 5 et 6 peuvent desormais fonctionner hors-ligne.")
        return 0
    except Exception as exc:
        print("ECHEC du telechargement :", exc)
        print(
            "L'application reste utilisable : l'Agent 4 basculera automatiquement "
            "sur le repli statistique LSA (TF-IDF + SVD), disponible hors-ligne."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
