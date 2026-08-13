"""
Agent 5 — Recherche
====================

Role (rapport de conception, section 3.3.2) : interroger la base de
connaissances de referentiels pedagogiques etrangers, indexee dans une base
vectorielle (FAISS), afin d'y retrouver, pour chaque unite de contenu du
cours, les passages les plus proches semantiquement.

Nature de l'agent : **deterministe**, mais operant sur les representations
apprises par l'Agent 4. Concretement :

1. les vecteurs des notions des referentiels sont charges dans un index FAISS
   `IndexFlatIP` (produit scalaire sur vecteurs normalises = similarite
   cosinus exacte) ;
2. chaque unite du cours interroge l'index et recupere ses `k` plus proches
   voisins ;
3. la recherche est ensuite **inversee** : pour chaque notion du referentiel,
   on conserve l'unite de cours qui lui ressemble le plus. C'est cette vue
   « par notion » qui alimente directement la cartographie de couverture
   produite par l'Agent 6.

Entree : sorties des Agents 3 et 4 + notions des referentiels
Sortie : voisinages semantiques dans les deux sens (voir `process()`)
"""

from collections import defaultdict

from app.services.vector_store import VectorIndex

TOP_K = 5


def process(agent3: dict, agent4: dict, notions_reference: list[dict], top_k: int = TOP_K) -> dict:
    """
    Execute l'Agent 5.

    Retourne :
    {
        "index": {...infos base vectorielle...},
        "top_k": int,
        "voisins_par_unite": [{"unite_id", "chapitre", "extrait", "voisins": [...]}],
        "meilleure_unite_par_notion": {
            "<code_pays>::<notion>": {"score", "unite_id", "chapitre", "extrait"}
        },
        "couverture_brute_par_pays": {"<code>": [{"notion", "score", ...}]}
    }
    """
    unites = agent3.get("unites", [])
    vecteurs_cours = agent4["_vecteurs_cours"]
    vecteurs_reference = agent4["_vecteurs_referentiel"]

    index = VectorIndex(agent4["dimension"])
    if len(notions_reference):
        index.ajouter(vecteurs_reference, notions_reference)

    # ------------------------------------------------------------------
    # Sens 1 : cours -> referentiels (k plus proches notions par unite)
    # ------------------------------------------------------------------
    voisins_par_unite = []
    if len(unites) and index.taille:
        resultats = index.rechercher(vecteurs_cours, k=top_k)
        for unite, voisins in zip(unites, resultats):
            voisins_par_unite.append(
                {
                    "unite_id": unite["id"],
                    "chapitre": unite["chapitre"],
                    "page": unite.get("page"),
                    "extrait": unite["texte"][:220],
                    "voisins": [
                        {
                            "pays": v["pays"],
                            "code": v["code"],
                            "drapeau": v.get("drapeau", ""),
                            "notion": v["notion"],
                            "score": v["score"],
                        }
                        for v in voisins
                    ],
                }
            )

    # ------------------------------------------------------------------
    # Sens 2 : referentiels -> cours (meilleure unite par notion)
    # ------------------------------------------------------------------
    index_cours = VectorIndex(agent4["dimension"])
    metadonnees_cours = [
        {
            "unite_id": u["id"],
            "chapitre": u["chapitre"],
            "page": u.get("page"),
            "extrait": u["texte"][:220],
        }
        for u in unites
    ]
    if len(unites):
        index_cours.ajouter(vecteurs_cours, metadonnees_cours)

    meilleure_unite_par_notion: dict[str, dict] = {}
    couverture_par_pays: dict[str, list] = defaultdict(list)

    if index_cours.taille and len(notions_reference):
        # On conserve plusieurs unites candidates par notion : l'Agent 6 les
        # re-score ensuite avec un cross-encodeur, qui evalue conjointement la
        # paire (unite, notion) la ou le bi-encodeur les a encodees separement.
        k_inverse = min(max(3, top_k), index_cours.taille)
        resultats_inverses = index_cours.rechercher(vecteurs_reference, k=k_inverse)
        for notion, voisins in zip(notions_reference, resultats_inverses):
            cle = f"{notion['code']}::{notion['notion']}"
            meilleur = voisins[0] if voisins else None
            entree = {
                "notion": notion["notion"],
                "descriptif": notion.get("descriptif", ""),
                "pays": notion["pays"],
                "code": notion["code"],
                "drapeau": notion.get("drapeau", ""),
                "referentiel": notion.get("referentiel", ""),
                "score": meilleur["score"] if meilleur else 0.0,
                "unite_id": meilleur["unite_id"] if meilleur else None,
                "chapitre": meilleur["chapitre"] if meilleur else None,
                "extrait": meilleur["extrait"] if meilleur else "",
                "unites_candidates": [
                    {
                        "unite_id": v["unite_id"],
                        "chapitre": v["chapitre"],
                        "page": v.get("page"),
                        "score": v["score"],
                        "extrait": v["extrait"],
                    }
                    for v in voisins
                ],
                "autres_correspondances": [
                    {"chapitre": v["chapitre"], "score": v["score"]} for v in voisins[1:]
                ],
            }
            meilleure_unite_par_notion[cle] = entree
            couverture_par_pays[notion["code"]].append(entree)

    return {
        "index": index.infos(),
        "top_k": top_k,
        "nb_unites_interrogees": len(unites),
        "nb_notions_indexees": index.taille,
        "voisins_par_unite": voisins_par_unite,
        "meilleure_unite_par_notion": meilleure_unite_par_notion,
        "couverture_brute_par_pays": dict(couverture_par_pays),
    }
