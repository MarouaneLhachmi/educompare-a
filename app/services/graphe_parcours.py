"""
Ordonnancement d'un parcours par centralite de graphe.
=======================================================

Deuxieme algorithme local de priorisation, independant du planificateur par
renforcement de `services/rl_parcours`. Les deux repondent a la meme question
— « par quoi commencer ? » — mais ne la posent pas de la meme facon, et c'est
tout l'interet de les faire coexister.

    Planificateur RL                    Ordonnancement par graphe
    ------------------------------      ------------------------------
    Decision sequentielle sous          Analyse structurelle, statique
    incertitude (MDP + TD(0))
    « Quelle action maximise le         « Quelles notions debloquent le
    gain cumule attendu ? »             plus d'autres notions ? »
    Simule l'acquisition seance         Aucun modele d'eleve, aucune
    apres seance (modele type BKT)      simulation temporelle
    Entree decisive : maitrise          Entree decisive : topologie des
    estimee + budget de seances         prerequis + gravite de l'ecart

Ils peuvent donc **diverger reellement**. Le planificateur privilegie une
notion grave dont la remediation rapporte immediatement ; l'ordonnancement par
graphe privilegie une notion peut-etre moins grave, mais dont depend une
grappe entiere du programme. Ce desaccord n'est pas un defaut : c'est
l'information qu'une source unique ne peut pas produire.

Pourquoi PageRank et pas un modele appris
------------------------------------------
La contrainte de donnees du projet est connue : tres peu d'analyses, aucune
etiquette de priorite produite par un humain. Un modele multi-criteres
supervise supposerait donc des etiquettes inventees — exactement ce que le
projet s'interdit. PageRank est **non supervise** : il ne s'entraine sur rien,
il se calcule sur le graphe de prerequis deja construit par
`services/prerequis`. Le projet emploie d'ailleurs deja cette famille
d'algorithmes, avec TextRank dans le module Rapport et Restitution.

Trois etages
------------
1. **Centralite.** PageRank personnalise sur le graphe de prerequis inverse.
   Une arete va du prerequis vers la notion qui en depend ; en inversant le
   sens, le score remonte vers les notions **fondatrices**. La
   personnalisation injecte la gravite de l'ecart : une notion sans lacune
   n'a pas besoin d'etre priorisee, meme si elle est structurellement
   centrale.
2. **Ordonnancement.** Tri par score decroissant, puis **reordonnancement
   topologique** : une notion n'est jamais placee avant l'un de ses prerequis
   egalement retenu. Le score dit quoi traiter, le graphe dit dans quel ordre.
3. **Choix de l'intervention.** Deterministe, a partir du type d'ecart et de
   la maitrise — la meme table d'interventions que le planificateur RL, pour
   que les deux parcours restent comparables.
"""

import numpy as np

from app.services import rl_parcours

# Amortissement de PageRank. La valeur usuelle 0,85 est conservee : elle est
# documentee, comparable, et rien dans ce graphe ne justifie de s'en ecarter.
AMORTISSEMENT = 0.85
# Au-dela, l'iteration n'apporte plus de decimale utile sur des graphes de
# quelques dizaines de sommets.
MAX_ITERATIONS = 100
TOLERANCE = 1.0e-9

# Poids du score structurel face a la gravite mesuree de l'ecart. La
# centralite prime, sinon cet algorithme rendrait le meme verdict que le
# classement par gravite — et n'apporterait rien de plus que l'Agent 6.
POIDS_CENTRALITE = 0.65
POIDS_GRAVITE = 0.35

# En dessous, la notion est consideree acquise : la traiter reviendrait a
# faire reviser ce qui est deja su.
MAITRISE_ACQUISE = 0.75


def _pagerank(adjacence: np.ndarray, personnalisation: np.ndarray) -> np.ndarray:
    """
    PageRank personnalise, par iteration de la puissance.

    `adjacence[i, j] = 1` signifie « i pointe vers j ». Les sommets sans
    successeur (puits) redistribuent leur masse selon le vecteur de
    personnalisation plutot que uniformement : sans cela, une notion terminale
    du programme diffuserait son importance sur des notions sans rapport.
    """
    n = len(adjacence)
    if n == 0:
        return np.zeros(0)

    personnalisation = np.asarray(personnalisation, dtype="float64")
    total = personnalisation.sum()
    personnalisation = (personnalisation / total) if total > 0 else np.full(n, 1.0 / n)

    degres = adjacence.sum(axis=1)
    transition = np.zeros_like(adjacence, dtype="float64")
    non_puits = degres > 0
    transition[non_puits] = adjacence[non_puits] / degres[non_puits, None]

    scores = personnalisation.copy()
    for _ in range(MAX_ITERATIONS):
        masse_puits = scores[~non_puits].sum()
        suivant = (
            AMORTISSEMENT * (scores @ transition + masse_puits * personnalisation)
            + (1.0 - AMORTISSEMENT) * personnalisation
        )
        somme = suivant.sum()
        if somme > 0:
            suivant = suivant / somme
        if np.abs(suivant - scores).sum() < TOLERANCE:
            return suivant
        scores = suivant
    return scores


def _intervention_pour(type_ecart: str, maitrise: float, prerequis_acquis: float) -> dict:
    """
    Choix deterministe de l'intervention, sans apprentissage.

    L'ordre des tests porte la regle pedagogique : un prerequis manquant se
    traite avant la notion elle-meme, une notion absente demande un apport
    theorique, une notion deja abordee demande de l'entrainement plutot qu'un
    nouveau cours.
    """
    interventions = {i["cle"]: i for i in rl_parcours.INTERVENTIONS}

    if prerequis_acquis < 0.50 and "remediation_prerequis" in interventions:
        return interventions["remediation_prerequis"]
    if type_ecart == "absente":
        return interventions.get("theorie") or rl_parcours.INTERVENTIONS[0]
    if type_ecart in ("evoquee_non_enseignee", "amorcee"):
        return interventions.get("theorie") or rl_parcours.INTERVENTIONS[0]
    if type_ecart == "superficielle":
        return interventions.get("exercices") or rl_parcours.INTERVENTIONS[-1]
    if maitrise < 0.55:
        return interventions.get("exercices") or rl_parcours.INTERVENTIONS[-1]
    return interventions.get("evaluation") or rl_parcours.INTERVENTIONS[-1]


def _justifier(notion: dict, score: float, nb_dependantes: int,
               prerequis_manquant: str | None) -> str:
    morceaux = []
    if nb_dependantes >= 3:
        morceaux.append(
            f"notion structurante : {nb_dependantes} notions du programme en dépendent"
        )
    elif nb_dependantes >= 1:
        morceaux.append(f"{nb_dependantes} notion(s) du programme en dépendent")
    else:
        morceaux.append("notion terminale du programme, sans dépendance en aval")

    if notion["gravite"] >= 0.75:
        morceaux.append("écart diagnostiqué comme sévère")
    elif notion["gravite"] >= 0.45:
        morceaux.append("écart modéré")

    if prerequis_manquant:
        morceaux.append(f"le prérequis « {prerequis_manquant} » reste à consolider")

    return (
        f"Centralité {score:.3f} dans le graphe de prérequis — "
        + ", ".join(morceaux) + "."
    )


def planifier(notions_etat: list[dict], graphe: dict, max_etapes: int = 12) -> dict:
    """
    Construit un parcours de priorisation par centralite de graphe.

    Retourne la meme forme que `rl_parcours.planifier()` pour les champs que
    l'affichage et la confrontation consomment (`etapes`, `nb_etapes`,
    `disponible`), afin que les deux sources restent directement comparables.

    Ne leve jamais : un graphe vide ou des notions absentes produisent un
    resultat indisponible et motive, jamais une exception.
    """
    if not notions_etat:
        return {"disponible": False, "motif": "aucune notion de référentiel à ordonner",
                "etapes": [], "nb_etapes": 0}

    cles = [n["cle"] for n in notions_etat]
    position = {cle: i for i, cle in enumerate(cles)}
    n = len(cles)

    prerequis = graphe.get("prerequis") or {}

    # Graphe INVERSE : une arete va du prerequis vers ce qu'il conditionne, on
    # la retourne pour que le score remonte vers les notions fondatrices.
    adjacence = np.zeros((n, n), dtype="float64")
    nb_dependantes = np.zeros(n, dtype=int)
    for cle, amonts in prerequis.items():
        if cle not in position:
            continue
        for amont in amonts:
            if amont in position:
                adjacence[position[cle], position[amont]] = 1.0
                nb_dependantes[position[amont]] += 1

    gravite = np.array([float(x["gravite"]) for x in notions_etat])
    maitrise = np.array([float(x["maitrise"]) for x in notions_etat])
    consensus = np.array([float(x.get("consensus", 0.2)) for x in notions_etat])

    # Personnalisation : la centralite est ponderee par le besoin reel. Une
    # notion centrale mais deja maitrisee ne doit pas remonter.
    besoin = np.clip(gravite * (1.0 - maitrise), 0.0, 1.0)
    personnalisation = besoin + 0.05  # plancher : aucun sommet n'est exclu du calcul

    centralite = _pagerank(adjacence, personnalisation)
    if centralite.max() > 0:
        centralite_normalisee = centralite / centralite.max()
    else:
        centralite_normalisee = centralite

    score = (POIDS_CENTRALITE * centralite_normalisee
             + POIDS_GRAVITE * np.clip(gravite * (1.0 - maitrise), 0.0, 1.0))

    # --- Selection : on ecarte ce qui est deja acquis --------------------
    candidats = [
        i for i in range(n)
        if maitrise[i] < MAITRISE_ACQUISE and score[i] > 0
    ]
    if not candidats:
        return {
            "disponible": False,
            "motif": "aucune notion ne présente d'écart à traiter",
            "etapes": [], "nb_etapes": 0,
            "moteur": "PageRank personnalisé sur le graphe de prérequis",
        }

    candidats.sort(key=lambda i: -score[i])
    retenus = candidats[:max_etapes]

    # --- Reordonnancement topologique -------------------------------------
    # Le score dit QUOI traiter, le graphe dit DANS QUEL ORDRE : une notion ne
    # peut pas passer avant un de ses prerequis egalement retenu.
    ensemble = set(retenus)
    ordonnes: list[int] = []
    restants = list(retenus)
    while restants:
        libres = [
            i for i in restants
            if not any(position.get(a) in ensemble and position.get(a) not in
                       {j for j in ordonnes}
                       for a in prerequis.get(cles[i], []))
        ]
        if not libres:
            # Securite : un cycle ne devrait pas exister (le graphe est
            # acyclique par construction), mais on ne bloque jamais.
            libres = [max(restants, key=lambda i: score[i])]
        choisi = max(libres, key=lambda i: score[i])
        ordonnes.append(choisi)
        restants.remove(choisi)

    # --- Construction des etapes ------------------------------------------
    etapes = []
    seances = 0.0
    for rang, i in enumerate(ordonnes, start=1):
        source = notions_etat[i]

        amonts = [a for a in prerequis.get(cles[i], []) if a in position]
        prerequis_acquis = (
            float(np.mean([maitrise[position[a]] for a in amonts])) if amonts else 1.0
        )
        amont_faible = next(
            (notions_etat[position[a]]["notion"] for a in amonts
             if maitrise[position[a]] < 0.50),
            None,
        )

        intervention = _intervention_pour(
            source.get("type_ecart", ""), maitrise[i], prerequis_acquis
        )
        seances += intervention["cout_seances"]

        etapes.append({
            "rang": rang,
            "cle": source["cle"],
            "notion": source.get("notion", ""),
            "descriptif": source.get("descriptif", ""),
            "pays": source.get("pays", ""),
            "code": source.get("code", ""),
            "intervention": intervention["cle"],
            "intervention_nom": intervention["nom"],
            "intervention_description": intervention["description"],
            "bloom_cible": intervention["bloom_cible"],
            "cout_seances": intervention["cout_seances"],
            "seance_cumulee": round(seances, 1),
            "maitrise_avant": round(float(maitrise[i]), 3),
            "consensus": round(float(consensus[i]), 3),
            "gravite": round(float(gravite[i]), 3),
            "type_ecart": source.get("type_ecart", ""),
            "libelle_ecart": source.get("libelle_ecart", ""),
            # Propre a cet algorithme : ce que le RL ne calcule pas.
            "centralite": round(float(centralite_normalisee[i]), 4),
            "score_priorisation": round(float(score[i]), 4),
            "nb_notions_dependantes": int(nb_dependantes[i]),
            "prerequis_acquis": round(prerequis_acquis, 3),
            "prerequis_bloquant": bool(prerequis_acquis < 0.50),
            "justification": _justifier(
                source, float(centralite_normalisee[i]), int(nb_dependantes[i]),
                amont_faible,
            ),
        })

    return {
        "disponible": True,
        "moteur": "PageRank personnalisé sur le graphe de prérequis inversé",
        "nature": "Analyse structurelle de graphe (non supervisée)",
        "nb_etapes": len(etapes),
        "nb_notions_distinctes": len({e["cle"] for e in etapes}),
        "seances_planifiees": round(seances, 1),
        "etapes": etapes,
        "parametres": {
            "amortissement": AMORTISSEMENT,
            "poids_centralite": POIDS_CENTRALITE,
            "poids_gravite": POIDS_GRAVITE,
            "seuil_maitrise_acquise": MAITRISE_ACQUISE,
        },
        "graphe": {
            "methode": graphe.get("methode"),
            "nb_aretes": graphe.get("nb_aretes", 0),
            "nb_sommets": n,
        },
        "notion_la_plus_structurante": (
            max(etapes, key=lambda e: e["nb_notions_dependantes"])["notion"]
            if etapes else None
        ),
    }


def infos() -> dict:
    """Description du moteur, restituee dans la traçabilité du rapport."""
    return {
        "algorithme": "PageRank personnalisé (Page & Brin, 1998) sur le graphe de prérequis",
        "nature": "Analyse structurelle de graphe — non supervisée",
        "amortissement": AMORTISSEMENT,
        "entrainement": "aucun : le score se calcule sur le graphe, il ne s'apprend pas",
        "complementarite": (
            "Le planificateur par renforcement optimise un gain d'apprentissage "
            "simulé séance après séance ; cet algorithme mesure l'effet de "
            "levier structurel d'une notion sur le reste du programme. Les deux "
            "peuvent diverger, et c'est ce désaccord qui est informatif."
        ),
    }
