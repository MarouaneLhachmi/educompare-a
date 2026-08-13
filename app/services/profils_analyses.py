"""
Profils, trajectoire et recommandation sur l'historique des analyses.
======================================================================

Le Module Historique et Tableau de Bord se contentait de compter et de
moyenner : combien d'analyses, quelle note moyenne, quelle repartition par
matiere. Ces agregats decrivent le passe sans jamais l'interpreter. Trois
questions restaient sans reponse, alors qu'elles sont celles que se pose
l'utilisateur :

- **A quoi ressemblent mes cours ?** Les onze indicateurs de l'Agent 7
  forment un profil ; regrouper les analyses fait apparaitre des **types
  recurrents** — le cours complet mais superficiel, le cours court et dense,
  le cours lacunaire. `KMeans` les degage, le coefficient de silhouette
  choisit leur nombre plutot que de le fixer arbitrairement.

- **Est-ce que je progresse ?** Une regression lineaire sur la suite des
  notes donne une tendance et une projection, assorties de leur incertitude.
  Sur quatre analyses, l'honnetete impose de dire que la tendance n'est pas
  significative — c'est ce que fait le coefficient de determination.

- **Ai-je deja rencontre ce cas ?** Les `k plus proches voisins` dans
  l'espace des indicateurs retrouvent les analyses au profil comparable :
  ce qui a fonctionne pour l'une eclaire les autres.

Aucun modele de langage n'intervient : tout repose sur les indicateurs deja
mesures par les agents.
"""

import numpy as np

from app.agents.agent7_evaluation import INDICATEURS

# Les onze indicateurs de l'Agent 7 constituent l'espace de representation.
CLES_INDICATEURS = [cle for cle, _, _ in INDICATEURS]
LIBELLES_INDICATEURS = {cle: libelle for cle, libelle, _ in INDICATEURS}

# En dessous de ce nombre d'analyses, aucun regroupement n'a de sens.
MIN_ANALYSES_CLUSTERING = 6
MIN_ANALYSES_TENDANCE = 4

# Etiquettes des profils types, attribuees selon les indicateurs dominants du
# groupe. Elles rendent un numero de cluster interpretable.
GABARITS_PROFIL = [
    ("couverture_internationale", "approfondissement", "Large mais superficiel",
     "couvre beaucoup de notions, sans les traiter en profondeur"),
    ("approfondissement", "couverture_internationale", "Ciblé et approfondi",
     "traite solidement un périmètre restreint"),
    ("structuration", "profondeur_cognitive", "Bien structuré, peu exigeant",
     "progression claire mais qui ne dépasse pas la restitution"),
    ("profondeur_cognitive", "couverture_internationale", "Exigeant mais lacunaire",
     "fait analyser et créer, sur un périmètre incomplet"),
]


def _matrice(analyses: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Extrait la matrice des indicateurs des analyses exploitables."""
    lignes, retenues = [], []
    for analyse in analyses:
        indicateurs = (analyse.get("agent7") or {}).get("indicateurs") or {}
        if not all(cle in indicateurs for cle in CLES_INDICATEURS):
            continue
        lignes.append([float(indicateurs[cle]) for cle in CLES_INDICATEURS])
        retenues.append(analyse)
    if not lignes:
        return np.zeros((0, len(CLES_INDICATEURS))), []
    return np.asarray(lignes, dtype="float64"), retenues


def _nommer_profil(centre: np.ndarray, moyenne: np.ndarray) -> tuple[str, str]:
    """
    Nomme un groupe a partir de ce qui le distingue de la moyenne generale.

    On cherche le gabarit dont l'indicateur fort est le plus au-dessus de la
    moyenne et l'indicateur faible le plus en dessous. A defaut, on qualifie
    par le seul indicateur le plus saillant.
    """
    index = {cle: i for i, cle in enumerate(CLES_INDICATEURS)}
    ecarts = centre - moyenne

    meilleur, marge = None, 0.0
    for cle_forte, cle_faible, nom, description in GABARITS_PROFIL:
        score = ecarts[index[cle_forte]] - ecarts[index[cle_faible]]
        if score > marge:
            meilleur, marge = (nom, description), score

    if meilleur and marge > 0.10:
        return meilleur

    dominant = CLES_INDICATEURS[int(np.argmax(np.abs(ecarts)))]
    sens = "élevé" if ecarts[index[dominant]] > 0 else "faible"
    return (
        f"{LIBELLES_INDICATEURS[dominant]} — {sens}",
        f"se distingue surtout par un niveau {sens} sur cet indicateur",
    )


def regrouper(analyses: list[dict]) -> dict:
    """
    Degage les profils types parmi les analyses d'un utilisateur.

    Le nombre de groupes n'est pas impose : on evalue chaque valeur possible
    par le coefficient de silhouette et on retient celle qui separe le mieux.
    Une silhouette faible signifie qu'il n'y a pas de structure — le module le
    dit plutot que d'inventer des categories.
    """
    matrice, retenues = _matrice(analyses)
    if len(matrice) < MIN_ANALYSES_CLUSTERING:
        return {
            "applique": False,
            "motif": (
                f"{len(matrice)} analyse(s) exploitable(s) — minimum "
                f"{MIN_ANALYSES_CLUSTERING} pour dégager des profils"
            ),
            "profils": [],
        }

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    meilleur = {"silhouette": -1.0, "k": 2, "etiquettes": None}
    for k in range(2, min(5, len(matrice) - 1) + 1):
        try:
            etiquettes = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrice)
            score = float(silhouette_score(matrice, etiquettes))
        except Exception:
            continue
        if score > meilleur["silhouette"]:
            meilleur = {"silhouette": score, "k": k, "etiquettes": etiquettes}

    if meilleur["etiquettes"] is None:
        return {"applique": False, "motif": "regroupement non concluant", "profils": []}

    etiquettes = meilleur["etiquettes"]
    moyenne = matrice.mean(axis=0)
    profils = []
    for groupe in range(meilleur["k"]):
        masque = etiquettes == groupe
        centre = matrice[masque].mean(axis=0)
        nom, description = _nommer_profil(centre, moyenne)
        membres = [retenues[i] for i in np.where(masque)[0]]
        notes = [float(m.get("resume_note_globale") or 0) for m in membres]

        profils.append({
            "groupe": int(groupe),
            "nom": nom,
            "description": description,
            "nb_analyses": int(masque.sum()),
            "note_moyenne": round(float(np.mean(notes)), 1) if notes else 0.0,
            "indicateurs": {
                cle: round(float(centre[i]), 3) for i, cle in enumerate(CLES_INDICATEURS)
            },
            "ecarts_marquants": sorted(
                [
                    {"cle": cle, "libelle": LIBELLES_INDICATEURS[cle],
                     "ecart": round(float(centre[i] - moyenne[i]), 3)}
                    for i, cle in enumerate(CLES_INDICATEURS)
                ],
                key=lambda e: abs(e["ecart"]), reverse=True,
            )[:3],
            "analyses": [
                {"id": m.get("id"), "titre": m.get("titre_cours") or m.get("nom_fichier"),
                 "note": m.get("resume_note_globale")}
                for m in membres[:6]
            ],
        })

    profils.sort(key=lambda p: p["nb_analyses"], reverse=True)
    return {
        "applique": True,
        "algorithme": "KMeans, nombre de groupes choisi par coefficient de silhouette",
        "nb_groupes": meilleur["k"],
        "silhouette": round(meilleur["silhouette"], 3),
        "structure_nette": meilleur["silhouette"] >= 0.25,
        "nb_analyses": len(matrice),
        "profils": profils,
    }


def trajectoire(analyses: list[dict]) -> dict:
    """
    Tendance des notes dans le temps et projection de la prochaine.

    La regression est volontairement lineaire : sur une dizaine de points,
    tout modele plus riche capterait du bruit. Le coefficient de
    determination est restitue tel quel — c'est lui qui dit si la tendance
    merite d'etre lue.

    La serie est restreinte aux analyses partageant la **meme version de
    referentiel** que la plus recente. Une note ne veut rien dire hors du
    socle de connaissances qui l'a produite : melanger deux versions ferait
    passer une revision du referentiel pour une evolution du travail de
    l'enseignant.
    """
    ordonnees = sorted(
        [a for a in analyses if a.get("statut") == "TERMINEE"],
        key=lambda a: str(a.get("date_creation_iso") or ""),
    )

    version_reference = ordonnees[-1].get("referentiel_version") if ordonnees else None
    comparables = [
        a for a in ordonnees if a.get("referentiel_version") == version_reference
    ]
    nb_ecartees = len(ordonnees) - len(comparables)
    ordonnees = comparables

    notes = [float(a.get("resume_note_globale") or 0) for a in ordonnees]

    if len(notes) < MIN_ANALYSES_TENDANCE:
        return {
            "applique": False,
            "motif": (
                f"{len(notes)} analyse(s) — minimum {MIN_ANALYSES_TENDANCE} "
                f"pour estimer une tendance"
            ),
            "version_referentiel": version_reference,
            "nb_ecartees_autre_version": nb_ecartees,
            "points": [
                {"rang": i + 1, "note": n, "date": a.get("date_creation")}
                for i, (n, a) in enumerate(zip(notes, ordonnees))
            ],
        }

    x = np.arange(len(notes), dtype="float64")
    y = np.asarray(notes, dtype="float64")
    pente, ordonnee = np.polyfit(x, y, 1)
    prediction = pente * x + ordonnee

    residus = y - prediction
    variance = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum(residus ** 2)) / variance if variance > 1e-9 else 0.0
    ecart_type = float(np.std(residus))

    prochaine = float(np.clip(pente * len(notes) + ordonnee, 0, 100))
    significative = bool(r2 >= 0.30 and abs(pente) >= 0.8)

    return {
        "applique": True,
        "algorithme": "régression linéaire sur la suite des notes",
        "nb_points": len(notes),
        "version_referentiel": version_reference,
        "nb_ecartees_autre_version": nb_ecartees,
        "pente_par_analyse": round(float(pente), 2),
        "r2": round(r2, 3),
        "ecart_type_residus": round(ecart_type, 2),
        "significative": significative,
        "sens": "progression" if pente > 0 else ("recul" if pente < 0 else "stable"),
        "note_prochaine_predite": round(prochaine, 1),
        "intervalle_prochaine": [
            round(max(0.0, prochaine - 1.96 * ecart_type), 1),
            round(min(100.0, prochaine + 1.96 * ecart_type), 1),
        ],
        "lecture": (
            f"Vos notes progressent de {pente:+.1f} point par analyse."
            if significative and pente > 0 else
            f"Vos notes reculent de {abs(pente):.1f} point par analyse."
            if significative and pente < 0 else
            "Aucune tendance nette ne se dégage : les écarts entre analyses "
            "s'expliquent davantage par la diversité des supports que par une "
            "évolution."
        ),
        "points": [
            {"rang": i + 1, "note": n, "predite": round(float(prediction[i]), 1),
             "date": a.get("date_creation"), "id": a.get("id")}
            for i, (n, a) in enumerate(zip(notes, ordonnees))
        ],
    }


def analyses_similaires(analyse: dict, corpus: list[dict], k: int = 3) -> dict:
    """
    Analyses au profil d'indicateurs le plus proche.

    Utile a deux titres : retrouver un cas deja traite, et verifier qu'une
    note surprenante n'est pas isolee.
    """
    reference = (analyse.get("agent7") or {}).get("indicateurs") or {}
    if not all(cle in reference for cle in CLES_INDICATEURS):
        return {"applique": False, "motif": "indicateurs indisponibles", "voisins": []}

    autres = [a for a in corpus if a.get("id") != analyse.get("id")]
    matrice, retenues = _matrice(autres)
    if len(matrice) < 2:
        return {"applique": False, "motif": "trop peu d'analyses comparables", "voisins": []}

    from sklearn.neighbors import NearestNeighbors

    vecteur = np.asarray([[float(reference[cle]) for cle in CLES_INDICATEURS]])
    nb = min(k, len(matrice))
    modele = NearestNeighbors(n_neighbors=nb, metric="euclidean").fit(matrice)
    distances, indices = modele.kneighbors(vecteur)

    # Distance maximale possible dans un espace de 11 dimensions bornees [0,1].
    distance_max = float(np.sqrt(len(CLES_INDICATEURS)))
    voisins = []
    for distance, index in zip(distances[0], indices[0]):
        voisine = retenues[int(index)]
        voisins.append({
            "id": voisine.get("id"),
            "titre": voisine.get("titre_cours") or voisine.get("nom_fichier"),
            "matiere": voisine.get("matiere"),
            "note": voisine.get("resume_note_globale"),
            "date": voisine.get("date_creation"),
            "distance": round(float(distance), 3),
            "proximite_pct": round(100 * (1 - float(distance) / distance_max), 1),
        })

    return {
        "applique": True,
        "algorithme": "k plus proches voisins dans l'espace des 11 indicateurs",
        "nb_compares": len(matrice),
        "voisins": voisins,
    }


def analyses_aberrantes(analyses: list[dict]) -> dict:
    """
    Analyses dont le profil s'ecarte nettement des autres.

    Une note isolee peut signaler un document mal extrait ou une matiere mal
    declaree — un incident a verifier plutot qu'un resultat a exploiter.
    """
    matrice, retenues = _matrice(analyses)
    if len(matrice) < 8:
        return {"applique": False, "motif": "trop peu d'analyses", "aberrantes": []}

    from sklearn.ensemble import IsolationForest

    modele = IsolationForest(n_estimators=150, contamination=0.12, random_state=42)
    predictions = modele.fit_predict(matrice)
    scores = modele.decision_function(matrice)

    aberrantes = [
        {
            "id": retenues[i].get("id"),
            "titre": retenues[i].get("titre_cours") or retenues[i].get("nom_fichier"),
            "note": retenues[i].get("resume_note_globale"),
            "date": retenues[i].get("date_creation"),
            "score": round(float(scores[i]), 4),
        }
        for i in range(len(matrice)) if predictions[i] == -1
    ]
    aberrantes.sort(key=lambda a: a["score"])

    return {
        "applique": True,
        "algorithme": "IsolationForest sur le profil d'indicateurs",
        "nb_analyses": len(matrice),
        "nb_aberrantes": len(aberrantes),
        "aberrantes": aberrantes[:6],
    }
