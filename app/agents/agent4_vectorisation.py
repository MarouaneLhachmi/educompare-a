"""
Agent 4 — Vectorisation
========================

Role (rapport de conception, section 3.3.2) : transformer chaque unite de
contenu en une representation numerique (vectorielle), a l'aide de modeles de
type *sentence-transformers*, permettant une comparaison mathematique du
contenu du cours avec les referentiels etrangers.

Nature de l'agent : **apprentissage automatique**. Deux moteurs sont
disponibles, selectionnes automatiquement :

1. **Deep Learning — sentence-transformers** (moteur nominal)
   Modele `paraphrase-multilingual-MiniLM-L12-v2` : un Transformer distille
   (12 couches, 384 dimensions) entraine par apprentissage contrastif sur des
   paires de phrases paralleles dans plus de 50 langues. Il projette une
   phrase francaise et sa traduction anglaise en des points voisins de
   l'espace vectoriel — propriete indispensable ici, puisque le cours est en
   francais alors que plusieurs referentiels sont rediges en anglais.

2. **Machine Learning — LSA (TF-IDF + SVD tronquee)** (repli hors-ligne)
   Si le modele neuronal n'a pas pu etre telecharge, l'agent bascule sur une
   analyse semantique latente : ponderation TF-IDF du corpus, puis reduction
   de dimension par decomposition en valeurs singulieres. Le resultat est
   moins performant en cross-lingue mais reste un espace vectoriel dense
   exploitable par la base FAISS.

Entree : unites de l'Agent 3 + notions des referentiels a indexer
Sortie : matrices de vecteurs + description du moteur utilise
"""

import threading

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from app.config import Config

_MODELE = None
_MODELE_TESTE = False
_MODELE_ERREUR: str | None = None
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Moteur 1 : Transformer multilingue (Deep Learning)
# ---------------------------------------------------------------------------

def charger_modele_transformer():
    """Charge le modele sentence-transformers une seule fois par processus."""
    global _MODELE, _MODELE_TESTE, _MODELE_ERREUR
    if _MODELE_TESTE:
        return _MODELE
    with _LOCK:
        if _MODELE_TESTE:
            return _MODELE
        _MODELE_TESTE = True
        if not Config.USE_EMBEDDINGS:
            _MODELE_ERREUR = "Desactive par configuration (USE_EMBEDDINGS=false)."
            return None
        try:
            from sentence_transformers import SentenceTransformer

            _MODELE = SentenceTransformer(Config.EMBEDDING_MODEL)
        except Exception as exc:
            _MODELE = None
            _MODELE_ERREUR = str(exc)[:200]
    return _MODELE


def _dimension_du_modele(modele) -> int:
    """
    Dimension de l'espace d'embedding. La methode a ete renommee entre deux
    versions majeures de sentence-transformers : on interroge les deux noms,
    avec un dernier recours par encodage d'une chaine temoin.
    """
    for nom in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        methode = getattr(modele, nom, None)
        if callable(methode):
            try:
                valeur = methode()
                if valeur:
                    return int(valeur)
            except Exception:
                continue
    return int(np.asarray(modele.encode(["dimension"])).shape[1])


class VectoriseurTransformer:
    """Encodeur neuronal multilingue."""

    type_moteur = "deep-learning"

    def __init__(self, modele):
        self._modele = modele
        self.nom = f"sentence-transformers / {Config.EMBEDDING_MODEL}"
        self.description = (
            "Transformer multilingue distille, entraine par apprentissage contrastif ; "
            "aligne les langues dans un espace vectoriel commun (comparaison FR/EN possible)."
        )
        self.dimension = int(_dimension_du_modele(modele))

    def encoder(self, textes: list[str]) -> np.ndarray:
        vecteurs = self._modele.encode(
            textes, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecteurs, dtype="float32")


class VectoriseurLSA:
    """Encodeur statistique de repli : TF-IDF + SVD tronquee (analyse semantique latente)."""

    type_moteur = "machine-learning"

    def __init__(self, corpus: list[str], dimension_cible: int = 256):
        self.nom = "LSA — TF-IDF + SVD tronquee (repli hors-ligne)"
        self.description = (
            "Ponderation TF-IDF (1-2 grammes) puis reduction de dimension par SVD. "
            "Repli active car le modele neuronal n'est pas disponible localement."
        )
        vectoriseur = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=20000
        )
        matrice = vectoriseur.fit_transform(corpus)
        n_composantes = max(2, min(dimension_cible, matrice.shape[1] - 1, len(corpus) - 1))
        self._pipeline = make_pipeline(
            vectoriseur, TruncatedSVD(n_components=n_composantes, random_state=42), Normalizer()
        )
        self._pipeline.fit(corpus)
        self.dimension = n_composantes

    def encoder(self, textes: list[str]) -> np.ndarray:
        return np.asarray(self._pipeline.transform(textes), dtype="float32")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def construire_vectoriseur(corpus_apprentissage: list[str]):
    """
    Retourne le meilleur encodeur disponible. `corpus_apprentissage` n'est
    utilise que par le repli LSA, qui doit etre ajuste sur les donnees.
    """
    modele = charger_modele_transformer()
    if modele is not None:
        return VectoriseurTransformer(modele)
    corpus = [t for t in corpus_apprentissage if t and t.strip()] or ["contenu"]
    return VectoriseurLSA(corpus)


def process(agent3: dict, notions_reference: list[dict]) -> dict:
    """
    Execute l'Agent 4.

    Retourne :
    {
        "moteur", "type_moteur", "description", "dimension",
        "nb_vecteurs_cours", "nb_vecteurs_referentiel", "repli_actif",
        "_vecteurs_cours": np.ndarray,        (non persiste)
        "_vecteurs_referentiel": np.ndarray,  (non persiste)
    }
    """
    textes_cours = [u["texte"] for u in agent3.get("unites", [])]
    textes_reference = [n["texte"] for n in notions_reference]

    vectoriseur = construire_vectoriseur(textes_cours + textes_reference)

    vecteurs_cours = vectoriseur.encoder(textes_cours) if textes_cours else np.zeros(
        (0, vectoriseur.dimension), dtype="float32"
    )
    vecteurs_reference = vectoriseur.encoder(textes_reference) if textes_reference else np.zeros(
        (0, vectoriseur.dimension), dtype="float32"
    )

    return {
        "moteur": vectoriseur.nom,
        "type_moteur": vectoriseur.type_moteur,
        "description": vectoriseur.description,
        "dimension": vectoriseur.dimension,
        "nb_vecteurs_cours": int(vecteurs_cours.shape[0]),
        "nb_vecteurs_referentiel": int(vecteurs_reference.shape[0]),
        "repli_actif": vectoriseur.type_moteur != "deep-learning",
        "motif_repli": _MODELE_ERREUR if vectoriseur.type_moteur != "deep-learning" else None,
        "_vecteurs_cours": vecteurs_cours,
        "_vecteurs_referentiel": vecteurs_reference,
        "_vectoriseur": vectoriseur,
    }
