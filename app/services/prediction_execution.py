"""
Prediction de la duree d'execution et detection d'executions atypiques.
========================================================================

La barre de progression du Module Traitement et Analyse reposait sur des
poids fixes : l'Agent 1 valait 8 %, l'Agent 6 en valait 20, quel que soit le
document. Ces valeurs avaient ete posees a la main, une fois, et ne
correspondaient a rien de mesure. Sur un cours de trois cents mots comme sur
un manuel de quatre-vingts pages, l'utilisateur voyait la meme progression —
donc une progression fausse dans les deux cas.

Ce module remplace ces constantes par une **mesure apprise**.

Prediction de duree
-------------------

Un `GradientBoostingRegressor` predit la duree de chaque agent a partir des
caracteristiques du document et de l'agent concerne. L'agent est encode
comme une variable explicative parmi d'autres, ce qui permet d'entrainer un
modele unique sur l'ensemble des observations plutot que neuf modeles
disposant chacun d'une poignee d'exemples — un choix impose par le volume de
donnees disponible.

Amorcage honnete : tant qu'aucune analyse n'a ete enregistree, le modele est
ajuste sur un jeu **synthetique** derive des anciens poids fixes, avec une
dispersion realiste. Il ne fait alors pas mieux que les constantes qu'il
remplace, et le back-office l'indique. Des que des analyses reelles existent,
elles prennent le dessus : le modele apprend ce que couter reellement veut
dire sur cette instance, avec ce materiel.

Detection d'executions atypiques
--------------------------------

Un `IsolationForest` apprend le profil normal de repartition des durees. Une
execution ou l'Agent 6 prend soudain trois fois sa part habituelle signale
un incident — cross-encodeur qui bascule sur le processeur, base vectorielle
degradee, document pathologique — que le seul temps total ne revelerait pas.
"""

import numpy as np

from app.services import entrainement

# Ordre fige des agents : il definit l'encodage de la variable « agent » et
# l'ordre des vecteurs de duree.
AGENTS = ["agent1", "agent2", "agent3", "agent4", "agent5", "agent6",
          "agent7", "agent8", "agent9"]
INDEX_AGENT = {cle: i for i, cle in enumerate(AGENTS)}

# Repartition historique, conservee comme *a priori* d'amorcage. Ce sont les
# anciennes constantes du module : elles ne disparaissent pas, elles
# deviennent une hypothese initiale que les donnees viennent corriger.
REPARTITION_APRIORI = {
    "agent1": 0.08, "agent2": 0.14, "agent3": 0.06, "agent4": 0.16,
    "agent5": 0.06, "agent6": 0.20, "agent7": 0.08, "agent8": 0.14,
    "agent9": 0.08,
}
DUREE_TOTALE_APRIORI = 95.0

CARACTERISTIQUES = ["nb_pages", "nb_mots", "nb_unites", "nb_notions",
                    "moteur_profond", "indice_agent"]


def caracteristiques_document(analyse: dict) -> dict:
    """
    Extrait les caracteristiques predictives connues d'une analyse.

    Appelee avant l'execution, elle s'appuie sur la pre-extraction du module
    de depot ; appelee apres l'Agent 3, elle dispose des valeurs exactes. Les
    deux cas sont volontairement supportes : la prediction se raffine au fil
    du pipeline plutot que d'attendre de tout savoir.
    """
    apercu = ((analyse.get("document") or {}).get("triage") or {}).get("apercu") or {}
    agent1 = analyse.get("agent1") or {}
    agent3 = analyse.get("agent3") or {}

    nb_pages = agent1.get("nb_pages") or apercu.get("nb_pages") or 1
    nb_mots = agent1.get("nb_mots") or apercu.get("nb_mots") or 200
    nb_unites = agent3.get("nb_unites") or max(1, round(nb_mots / 110))
    nb_notions = analyse.get("nb_notions_referentiel") or 40
    moteur_profond = 1.0
    if (analyse.get("agent4") or {}).get("type_moteur") == "machine-learning":
        moteur_profond = 0.0

    return {
        "nb_pages": float(nb_pages),
        "nb_mots": float(nb_mots),
        "nb_unites": float(nb_unites),
        "nb_notions": float(nb_notions),
        "moteur_profond": moteur_profond,
    }


def _vecteur(caracteristiques: dict, agent: str) -> list[float]:
    return [
        caracteristiques["nb_pages"],
        caracteristiques["nb_mots"],
        caracteristiques["nb_unites"],
        caracteristiques["nb_notions"],
        caracteristiques["moteur_profond"],
        float(INDEX_AGENT[agent]),
    ]


# ---------------------------------------------------------------------------
# Jeux d'apprentissage
# ---------------------------------------------------------------------------

def _observations_reelles() -> tuple[list[list[float]], list[float], int]:
    """Durees par agent effectivement mesurees lors des analyses passees."""
    X, y, nb_analyses = [], [], 0
    for analyse in entrainement.corpus_analyses():
        durees = analyse.get("durees_agents") or {}
        if not durees:
            continue
        nb_analyses += 1
        caracteristiques = caracteristiques_document(analyse)
        for agent, duree in durees.items():
            if agent in INDEX_AGENT and duree and duree > 0:
                X.append(_vecteur(caracteristiques, agent))
                y.append(float(duree))
    return X, y, nb_analyses


def _observations_synthetiques(n: int = 400, graine: int = 42):
    """
    Jeu d'amorcage derive de la repartition historique.

    La duree totale est tiree en fonction de la taille du document, puis
    repartie entre les agents selon les anciens poids, avec un bruit
    multiplicatif. Les agents qui appellent un modele de langage recoivent en
    outre une dispersion plus large : c'est leur comportement observe, la
    latence reseau dominant leur cout.
    """
    rng = np.random.default_rng(graine)
    X, y = [], []
    agents_reseau = {"agent2", "agent6", "agent8", "agent9"}

    for _ in range(n):
        nb_pages = float(rng.integers(1, 60))
        nb_mots = nb_pages * rng.uniform(120, 520)
        nb_unites = max(1.0, nb_mots / rng.uniform(90, 140))
        nb_notions = float(rng.integers(20, 90))
        moteur_profond = float(rng.random() > 0.15)

        # Le cout croit avec le volume, mais moins que proportionnellement :
        # une part du temps est incompressible (chargement, appels reseau).
        total = DUREE_TOTALE_APRIORI * (0.45 + 0.55 * (nb_unites / 40) ** 0.6)
        total *= rng.uniform(0.75, 1.35)
        if not moteur_profond:
            total *= 0.35  # le repli statistique est bien plus rapide

        caracteristiques = {
            "nb_pages": nb_pages, "nb_mots": nb_mots, "nb_unites": nb_unites,
            "nb_notions": nb_notions, "moteur_profond": moteur_profond,
        }
        for agent, part in REPARTITION_APRIORI.items():
            dispersion = 0.45 if agent in agents_reseau else 0.20
            duree = total * part * rng.lognormal(0, dispersion)
            if agent in {"agent4", "agent5", "agent6"}:
                duree *= (nb_notions / 55) ** 0.5  # cout indexe sur le referentiel
            X.append(_vecteur(caracteristiques, agent))
            y.append(max(0.05, float(duree)))
    return X, y


def _entrainer_duree() -> dict:
    from sklearn.ensemble import GradientBoostingRegressor

    X_reel, y_reel, nb_analyses = _observations_reelles()
    X_synth, y_synth = _observations_synthetiques()

    if len(y_reel) >= 45:  # au moins cinq analyses completes
        # Les observations reelles sont dupliquees pour peser davantage que
        # l'a priori synthetique, sans que celui-ci disparaisse : il continue
        # de couvrir les tailles de document jamais rencontrees.
        facteur = max(1, int(len(y_synth) / (2 * len(y_reel))))
        X = X_synth + X_reel * facteur
        y = y_synth + y_reel * facteur
        source = f"{nb_analyses} analyses réelles + a priori synthétique"
    else:
        X, y = X_synth, y_synth
        source = "a priori synthétique dérivé des anciens poids fixes (amorçage)"

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    decoupe = int(0.85 * len(X))

    modele = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42
    )
    modele.fit(X[:decoupe], y[:decoupe])

    prediction = modele.predict(X[decoupe:])
    erreur = float(np.mean(np.abs(prediction - y[decoupe:])))
    erreur_relative = float(np.mean(np.abs(prediction - y[decoupe:]) / np.maximum(y[decoupe:], 0.1)))

    return {
        "modele": modele,
        "source_donnees": source,
        "nb_observations": len(y_reel),
        "metrique_erreur_absolue_s": round(erreur, 2),
        "metrique_erreur_relative_pct": round(100 * erreur_relative, 1),
        "metrique_nb_analyses_reelles": nb_analyses,
    }


def _entrainer_anomalie() -> dict:
    """Profil normal de repartition des durees entre agents."""
    from sklearn.ensemble import IsolationForest

    profils = []
    for analyse in entrainement.corpus_analyses():
        durees = analyse.get("durees_agents") or {}
        vecteur = _profil_repartition(durees)
        if vecteur is not None:
            profils.append(vecteur)

    if len(profils) < 8:
        return {"modele": None, "source_donnees": "trop peu d'exécutions enregistrées",
                "nb_observations": len(profils)}

    modele = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
    modele.fit(np.asarray(profils))
    return {
        "modele": modele,
        "source_donnees": "répartition des durées des analyses passées",
        "nb_observations": len(profils),
    }


def _profil_repartition(durees: dict) -> list[float] | None:
    """
    Part de chaque agent dans la duree totale.

    On raisonne en parts et non en secondes absolues : une machine lente
    decale toutes les durees sans que la repartition change. C'est bien la
    deformation du profil qui signale un incident, pas la lenteur generale.
    """
    valeurs = [float(durees.get(agent) or 0.0) for agent in AGENTS]
    total = sum(valeurs)
    if total <= 0:
        return None
    return [v / total for v in valeurs]


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def predire_durees(analyse: dict) -> dict:
    """
    Duree prevue de chaque agent, en secondes.

    Retourne toujours un resultat exploitable : en cas d'indisponibilite du
    modele, la repartition historique sert de repli.
    """
    caracteristiques = caracteristiques_document(analyse)
    entraine = entrainement.obtenir("execution_duree", _entrainer_duree)
    modele = entraine.get("modele")

    if modele is None:
        durees = {a: DUREE_TOTALE_APRIORI * p for a, p in REPARTITION_APRIORI.items()}
        source = "répartition historique (modèle indisponible)"
    else:
        matrice = np.asarray([_vecteur(caracteristiques, a) for a in AGENTS])
        prediction = np.maximum(modele.predict(matrice), 0.05)
        durees = {a: float(d) for a, d in zip(AGENTS, prediction)}
        source = entraine.get("source_donnees")

    total = sum(durees.values())
    return {
        "durees": {a: round(d, 2) for a, d in durees.items()},
        "poids": {a: round(100 * d / total, 2) for a, d in durees.items()},
        "duree_totale_prevue_s": round(total, 1),
        "source": source,
        "amorcage": entraine.get("amorcage", True),
        "erreur_relative_pct": entraine.get("metrique_erreur_relative_pct"),
        "caracteristiques": caracteristiques,
    }


def detecter_anomalie(durees: dict) -> dict:
    """Compare le profil d'execution observe au profil normal appris."""
    profil = _profil_repartition(durees)
    if profil is None:
        return {"applique": False, "motif": "durées indisponibles"}

    entraine = entrainement.obtenir("execution_anomalie", _entrainer_anomalie)
    modele = entraine.get("modele")
    if modele is None:
        return {"applique": False, "motif": entraine.get("source_donnees"),
                "nb_observations": entraine.get("nb_observations", 0)}

    score = float(modele.decision_function([profil])[0])
    atypique = bool(modele.predict([profil])[0] == -1)

    # Agent dont la part s'ecarte le plus de la repartition attendue.
    ecarts = {
        agent: profil[i] - REPARTITION_APRIORI[agent]
        for i, agent in enumerate(AGENTS)
    }
    dominant = max(ecarts.items(), key=lambda kv: abs(kv[1]))

    return {
        "applique": True,
        "atypique": atypique,
        "score": round(score, 4),
        "modele": "IsolationForest sur la répartition des durées",
        "nb_observations": entraine.get("nb_observations", 0),
        "agent_le_plus_ecarte": dominant[0],
        "ecart_part": round(dominant[1], 3),
        "profil": {agent: round(profil[i], 3) for i, agent in enumerate(AGENTS)},
    }


def infos_modeles() -> list[dict]:
    return [e for e in entrainement.etat_modeles() if e["nom"].startswith("execution_")]
