"""
Base vectorielle (FAISS).
==========================

Section 3.4.2 du rapport : les representations vectorielles des referentiels
pedagogiques etrangers sont stockees dans une base vectorielle (FAISS),
interrogee par l'Agent 5 (Recherche) lors de chaque analyse.

Ce module encapsule FAISS derriere une interface minimale (`VectorIndex`).
Si la bibliotheque FAISS n'est pas installee sur la machine, un repli exact
en NumPy (produit scalaire sur vecteurs normalises, mathematiquement
equivalent a `IndexFlatIP`) prend le relais : les resultats sont identiques,
seule la vitesse differe sur de tres grands volumes.
"""

import numpy as np

try:  # pragma: no cover - depend de l'environnement d'installation
    import faiss

    FAISS_DISPONIBLE = True
except Exception:  # pragma: no cover
    faiss = None
    FAISS_DISPONIBLE = False


def normaliser(vecteurs: np.ndarray) -> np.ndarray:
    """Normalise chaque ligne (norme L2 = 1) pour que le produit scalaire
    corresponde exactement a la similarite cosinus."""
    vecteurs = np.asarray(vecteurs, dtype="float32")
    if vecteurs.ndim == 1:
        vecteurs = vecteurs.reshape(1, -1)
    normes = np.linalg.norm(vecteurs, axis=1, keepdims=True)
    normes[normes == 0] = 1.0
    return vecteurs / normes


class VectorIndex:
    """Index vectoriel de similarite cosinus, avec metadonnees associees."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.metadonnees: list[dict] = []
        self._vecteurs: np.ndarray | None = None
        if FAISS_DISPONIBLE:
            self._index = faiss.IndexFlatIP(dimension)
            self.moteur = "FAISS (IndexFlatIP)"
        else:
            self._index = None
            self.moteur = "NumPy (repli exact, FAISS non installe)"

    # ------------------------------------------------------------------
    def ajouter(self, vecteurs: np.ndarray, metadonnees: list[dict]) -> None:
        vecteurs = normaliser(vecteurs)
        if vecteurs.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension incompatible : index={self.dimension}, vecteurs={vecteurs.shape[1]}"
            )
        if self._index is not None:
            self._index.add(vecteurs)
        self._vecteurs = (
            vecteurs if self._vecteurs is None else np.vstack([self._vecteurs, vecteurs])
        )
        self.metadonnees.extend(metadonnees)

    # ------------------------------------------------------------------
    @property
    def taille(self) -> int:
        return len(self.metadonnees)

    # ------------------------------------------------------------------
    def rechercher(self, requetes: np.ndarray, k: int = 5) -> list[list[dict]]:
        """
        Pour chaque vecteur de requete, retourne les `k` entrees les plus
        proches sous la forme [{"score": float, **metadonnees}, ...].
        """
        if self.taille == 0:
            return [[] for _ in range(len(np.atleast_2d(requetes)))]

        requetes = normaliser(requetes)
        k = min(k, self.taille)

        if self._index is not None:
            scores, indices = self._index.search(requetes, k)
        else:
            produits = requetes @ self._vecteurs.T
            indices = np.argsort(-produits, axis=1)[:, :k]
            scores = np.take_along_axis(produits, indices, axis=1)

        resultats = []
        for ligne_scores, ligne_indices in zip(scores, indices):
            voisins = []
            for score, indice in zip(ligne_scores, ligne_indices):
                if indice < 0:
                    continue
                entree = dict(self.metadonnees[int(indice)])
                entree["score"] = round(float(score), 4)
                voisins.append(entree)
            resultats.append(voisins)
        return resultats

    # ------------------------------------------------------------------
    def infos(self) -> dict:
        return {
            "moteur": self.moteur,
            "dimension": self.dimension,
            "nb_vecteurs": self.taille,
            "faiss_disponible": FAISS_DISPONIBLE,
        }
