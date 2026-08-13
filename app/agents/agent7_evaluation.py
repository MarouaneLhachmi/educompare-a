"""
Agent 7 — Evaluation
=====================

Role (rapport de conception, section 3.3.2) : synthetiser les resultats de la
comparaison en un ensemble d'indicateurs d'evaluation de la qualite
pedagogique du cours, combinant criteres objectifs et qualitatifs.

Nature de l'agent : **apprentissage automatique supervise, non supervise et
profond**.

1. **Onze indicateurs mesures** (contre sept dans la version initiale), tous
   normalises sur [0, 1] et calcules a partir des sorties des Agents 1, 3, 4
   et 6. Les quatre nouveaux indicateurs repondent a une limite de fond de la
   premiere version : elle evaluait la *presence* des notions, jamais la
   *qualite d'apprentissage* que le support permet.

   - `approfondissement`      : part des notions reellement traitees en
                                profondeur, et non simplement mentionnees ;
   - `adequation_niveau`      : la difficulte reelle du texte correspond-elle
                                au niveau annonce ?
   - `progression_difficulte` : le support monte-t-il regulierement en
                                exigence, ou avance-t-il par a-coups ?
   - `profondeur_cognitive`   : jusqu'ou monte-t-il sur la taxonomie de
                                Bloom — se limite-t-il a restituer, ou fait-il
                                analyser et creer ?

2. **Coherence thematique — apprentissage non supervise.** Les vecteurs des
   unites de cours produits par l'Agent 4 sont regroupes par `KMeans`, la
   qualite du regroupement etant mesuree par le coefficient de silhouette. Un
   cours structure produit des groupes thematiques nets.

3. **Note globale — ensemble supervise par empilement (*stacking*).** Deux
   regresseurs aux biais complementaires sont combines par un meta-modele
   lineaire entraine en validation croisee :

   - `GradientBoostingRegressor` : arbres de decision boostes, excellent pour
     capter des effets de seuil (« en dessous de tel taux de couverture, la
     note s'effondre ») ;
   - `MLPRegressor` : reseau de neurones, qui capte au contraire les
     interactions continues et lisses entre indicateurs ;
   - `Ridge` comme meta-modele, qui apprend a doser les deux selon la region
     de l'espace des indicateurs.

   L'ecart entre les deux tetes est restitue comme **indicateur
   d'incertitude** : quand elles divergent, la note est moins fiable.

   Transparence methodologique : aucun corpus de cours annotes par des
   experts n'etant disponible a ce stade du projet, le modele est entraine
   sur un **jeu de donnees synthetique** genere a partir de la grille
   d'evaluation experte (ponderations, rendements decroissants, penalites
   d'incoherence, bruit gaussien). Cela valide la chaine d'apprentissage
   complete et permet de la reentrainer telle quelle sur des donnees reelles.
   Les indicateurs, eux, sont mesures sur le document — pas simules.

4. **Profil de maitrise par notion** — sortie nouvelle et centrale : pour
   chaque notion des referentiels, une estimation de la maitrise qu'un eleve
   suivant ce support peut en atteindre. C'est **l'etat initial du processus
   de decision markovien** exploite par l'Agent 8 pour planifier le parcours
   d'amelioration.

Entree : sorties des Agents 1, 2, 3, 4 et 6
Sortie : indicateurs, note globale, profil de maitrise, appreciation
"""

import threading

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from app.services import gemini_client, pedagogie

# ---------------------------------------------------------------------------
# Grille d'evaluation experte : ponderation de chacun des onze indicateurs
# ---------------------------------------------------------------------------
INDICATEURS = [
    ("couverture_internationale", "Couverture des référentiels internationaux", 0.20),
    ("approfondissement", "Profondeur de traitement effective des notions", 0.12),
    ("homogeneite_couverture", "Homogénéité entre les différents référentiels", 0.06),
    ("profondeur_contenu", "Volume de contenu par chapitre", 0.10),
    ("structuration", "Structuration et progression pédagogique", 0.10),
    ("richesse_pedagogique", "Richesse des dispositifs (exemples, exercices, évaluations)", 0.10),
    ("diversite_lexicale", "Diversité et précision du vocabulaire", 0.05),
    ("coherence_thematique", "Cohérence thématique interne du support", 0.07),
    ("adequation_niveau", "Adéquation de la difficulté au niveau visé", 0.08),
    ("progression_difficulte", "Régularité de la montée en difficulté", 0.06),
    ("profondeur_cognitive", "Profondeur cognitive (taxonomie de Bloom)", 0.06),
]

# Bandes de difficulte attendues par cycle, sur l'echelle de l'indice de
# lisibilite de `app/services/pedagogie.py`. Table de reference d'ingenierie,
# remplacee par le modele `niveau_reg` des qu'il est depose.
DIFFICULTE_ATTENDUE = {
    "primaire": (0.22, 0.45),
    "college": (0.38, 0.60),
    "lycee": (0.52, 0.74),
    "superieur": (0.62, 0.90),
}

_MODELE = None
_MODELE_INFOS: dict = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 1. Modele d'ensemble
# ---------------------------------------------------------------------------

def _generer_jeu_synthetique(n: int = 2000, graine: int = 42):
    """
    Genere un jeu d'entrainement synthetique a partir de la grille experte.

    La cible n'est volontairement pas une combinaison lineaire des
    indicateurs : on y injecte trois effets pedagogiques connus, afin que les
    modeles aient une structure non triviale a apprendre et que l'empilement
    ait un interet reel.

    - **Rendements decroissants** sur la couverture : passer de 20 % a 40 %
      de couverture vaut bien davantage que passer de 70 % a 90 %.
    - **Penalite d'incoherence** : un support mal structure ET peu approfondi
      se degrade plus vite que la somme de ses deux faiblesses.
    - **Effet de plafond cognitif** : un support qui ne depasse jamais la
      restitution voit sa note bornee, meme si tout le reste est excellent.
    """
    rng = np.random.default_rng(graine)
    X = rng.beta(a=2.2, b=2.0, size=(n, len(INDICATEURS)))
    poids = np.array([p for _, _, p in INDICATEURS])
    index = {cle: i for i, (cle, _, _) in enumerate(INDICATEURS)}

    base = X @ poids
    rendements = 0.10 * np.sqrt(np.clip(X[:, index["couverture_internationale"]], 0, 1))
    incoherence = -1.8 * (
        np.clip(0.40 - X[:, index["structuration"]], 0, None)
        * np.clip(0.40 - X[:, index["approfondissement"]], 0, None)
    )
    plafond = -0.12 * np.clip(0.30 - X[:, index["profondeur_cognitive"]], 0, None) / 0.30
    bruit = rng.normal(0, 0.025, size=n)

    y = np.clip(base + rendements + incoherence + plafond + bruit, 0, 1) * 100
    return X, y


def _obtenir_modele():
    """Entraine (une seule fois par processus) l'ensemble de regression."""
    global _MODELE, _MODELE_INFOS
    if _MODELE is not None:
        return _MODELE
    with _LOCK:
        if _MODELE is not None:
            return _MODELE

        X, y = _generer_jeu_synthetique()
        decoupe = int(0.8 * len(X))
        X_app, y_app = X[:decoupe], y[:decoupe]
        X_val, y_val = X[decoupe:], y[decoupe:]

        gbr = GradientBoostingRegressor(
            n_estimators=220, max_depth=3, learning_rate=0.07, random_state=42
        )
        # Le perceptron travaille sur des donnees centrees-reduites : sans
        # cela la descente de gradient converge mal.
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=3e-3,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=42,
            ),
        )
        ensemble = StackingRegressor(
            estimators=[("gbr", gbr), ("mlp", mlp)],
            final_estimator=Ridge(alpha=1.0),
            cv=5,           # predictions hors-echantillon pour le meta-modele
            passthrough=False,
        )
        ensemble.fit(X_app, y_app)

        def evaluer(modele):
            prediction = modele.predict(X_val)
            return (
                float(np.mean(np.abs(prediction - y_val))),
                float(1 - np.sum((y_val - prediction) ** 2) / np.sum((y_val - y_val.mean()) ** 2)),
            )

        mae_ens, r2_ens = evaluer(ensemble)
        mae_gbr, r2_gbr = evaluer(ensemble.named_estimators_["gbr"])
        mae_mlp, r2_mlp = evaluer(ensemble.named_estimators_["mlp"])

        _MODELE = ensemble
        _MODELE_INFOS = {
            "algorithme": "StackingRegressor — GradientBoosting + MLP, méta-modèle Ridge",
            "tetes": [
                {"nom": "GradientBoostingRegressor", "type": "Machine Learning (arbres boostés)",
                 "erreur_absolue_moyenne": round(mae_gbr, 2), "r2_validation": round(r2_gbr, 3)},
                {"nom": "MLPRegressor (64, 32)", "type": "Deep Learning (perceptron multicouche)",
                 "erreur_absolue_moyenne": round(mae_mlp, 2), "r2_validation": round(r2_mlp, 3)},
            ],
            "meta_modele": "Ridge (validation croisée à 5 plis)",
            "nb_indicateurs": len(INDICATEURS),
            "taille_entrainement": decoupe,
            "taille_validation": len(X) - decoupe,
            "erreur_absolue_moyenne": round(mae_ens, 2),
            "r2_validation": round(r2_ens, 3),
            "corpus": "jeu synthétique dérivé de la grille d'évaluation experte",
        }
    return _MODELE


def infos_modele() -> dict:
    _obtenir_modele()
    return dict(_MODELE_INFOS)


# ---------------------------------------------------------------------------
# 2. Mesures
# ---------------------------------------------------------------------------

def _coherence_thematique(agent4: dict) -> tuple[float, dict]:
    """Regroupement non supervise des unites de cours (KMeans + silhouette)."""
    vecteurs = agent4.get("_vecteurs_cours")
    if vecteurs is None or len(vecteurs) < 4:
        return 0.5, {"applique": False, "motif": "trop peu d'unités de contenu"}

    meilleur_score, meilleur_k = -1.0, 2
    for k in range(2, min(6, len(vecteurs) - 1) + 1):
        try:
            etiquettes = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vecteurs)
            score = float(silhouette_score(vecteurs, etiquettes))
        except Exception:
            continue
        if score > meilleur_score:
            meilleur_score, meilleur_k = score, k

    if meilleur_score < 0:
        return 0.5, {"applique": False, "motif": "regroupement non concluant"}

    return float(np.clip((meilleur_score + 1) / 2, 0, 1)), {
        "applique": True,
        "algorithme": "KMeans + coefficient de silhouette",
        "nb_groupes_optimal": meilleur_k,
        "silhouette": round(meilleur_score, 3),
    }


def _cycle_du_niveau(niveau: str) -> str:
    libelle = (niveau or "").lower()
    if any(mot in libelle for mot in ("primaire", "élémentaire", "elementaire", "cm1", "cm2")):
        return "primaire"
    if any(mot in libelle for mot in ("collège", "college", "6e", "5e", "4e", "3e")):
        return "college"
    if any(mot in libelle for mot in ("lycée", "lycee", "seconde", "première", "premiere", "terminale", "bac")):
        return "lycee"
    if any(mot in libelle for mot in ("licence", "master", "supérieur", "superieur", "universit")):
        return "superieur"
    return "primaire"


def _analyse_difficulte(agent3: dict, agent4: dict, niveau: str) -> dict:
    """Difficulte de chaque unite, adequation au niveau vise et progression."""
    unites = agent3.get("unites", [])
    textes = [u["texte"] for u in unites]
    estimation = pedagogie.estimer_niveaux(textes, agent4.get("_vecteurs_cours"))
    difficultes = estimation["difficultes"]

    if not difficultes:
        return {
            "difficultes": [], "mediane": 0.5, "cycle": _cycle_du_niveau(niveau),
            "adequation": 0.5, "progression": pedagogie.progression_difficulte([]),
            "source": estimation["source"], "ecart_au_cycle": None,
        }

    cycle = _cycle_du_niveau(niveau)
    borne_basse, borne_haute = DIFFICULTE_ATTENDUE[cycle]
    mediane = float(np.median(difficultes))

    if borne_basse <= mediane <= borne_haute:
        ecart = 0.0
    else:
        ecart = min(abs(mediane - borne_basse), abs(mediane - borne_haute))
    adequation = float(np.clip(1.0 - ecart / 0.35, 0.0, 1.0))

    return {
        "difficultes": difficultes,
        "par_unite": [
            {"unite_id": u["id"], "chapitre": u["chapitre"], "difficulte": d}
            for u, d in zip(unites, difficultes)
        ],
        "mediane": round(mediane, 3),
        "cycle": cycle,
        "bande_attendue": [borne_basse, borne_haute],
        "ecart_au_cycle": round(ecart, 3),
        "adequation": round(adequation, 3),
        "progression": pedagogie.progression_difficulte(difficultes),
        "source": estimation["source"],
    }


def _bloom_par_chapitre(agent1: dict, agent2: dict) -> dict:
    """Profil cognitif de chaque chapitre : objectifs declares + contenu."""
    objectifs = {
        c.get("titre", "").strip().lower(): c.get("objectifs_pedagogiques", [])
        for c in agent2.get("chapitres", [])
    }
    profils = {}
    for chapitre in agent1.get("chapitres", []):
        titre = chapitre.get("titre", "")
        enonces = list(objectifs.get(titre.strip().lower(), []))
        enonces.append(chapitre.get("contenu") or chapitre.get("extrait") or "")
        profils[titre] = pedagogie.profil_bloom(enonces)
    return profils


def _profil_maitrise(agent6: dict, profils_bloom: dict) -> dict:
    """
    Estime, pour chaque notion des referentiels, la maitrise qu'un eleve
    suivant ce support peut raisonnablement en atteindre.

    Formulation retenue :

        maitrise = p × (0,50 + 0,35 × suffisance + 0,15 × profondeur_cognitive)

    ou `p` est la probabilite de couverture mesuree par l'Agent 6. La
    multiplication par `p` est structurante : une notion absente du support ne
    peut pas etre maitrisee, quelles que soient les qualites du reste du
    document. A l'inverse, une notion parfaitement couverte mais traitee
    superficiellement et sans exigence cognitive plafonne a 0,50.

    Avertissement : il s'agit de la maitrise **attendue d'un eleve suivant ce
    support**, deduite des seules caracteristiques du document. Ce n'est en
    aucun cas la mesure de la maitrise d'un eleve reel, qui supposerait des
    donnees d'evaluation dont le systeme ne dispose pas.
    """
    entrees = []
    for pays in (agent6.get("par_pays") or {}).values():
        for notion in pays["notions"]:
            profil = profils_bloom.get(notion.get("chapitre_correspondant") or "", {})
            profondeur = float(profil.get("profondeur_cognitive", 0.0) or 0.0)
            p = float(notion["probabilite_couverture"])
            suffisance = float(notion.get("suffisance", 0.0))

            maitrise = p * (0.50 + 0.35 * suffisance + 0.15 * profondeur)
            entrees.append({
                "cle": f"{notion['code']}::{notion['notion']}",
                "code": notion["code"],
                "pays": notion["pays"],
                "notion": notion["notion"],
                "descriptif": notion.get("descriptif", ""),
                "maitrise": round(float(np.clip(maitrise, 0.0, 1.0)), 3),
                "probabilite_couverture": p,
                "suffisance": round(suffisance, 3),
                "profondeur_cognitive_chapitre": round(profondeur, 3),
                "bloom_chapitre": profil.get("niveau_max"),
                "statut": notion["statut"],
                "type_ecart": notion["type_ecart"],
                "libelle_ecart": notion.get("libelle_ecart", ""),
                "chapitre_correspondant": notion.get("chapitre_correspondant"),
            })

    entrees.sort(key=lambda e: e["maitrise"])
    maitrises = [e["maitrise"] for e in entrees]
    return {
        "par_notion": entrees,
        "nb_notions": len(entrees),
        "maitrise_globale": round(float(np.mean(maitrises)), 3) if maitrises else 0.0,
        "maitrise_mediane": round(float(np.median(maitrises)), 3) if maitrises else 0.0,
        "nb_maitrise_faible": sum(1 for m in maitrises if m < 0.35),
        "nb_maitrise_solide": sum(1 for m in maitrises if m >= 0.65),
        "formule": "maitrise = p × (0,50 + 0,35 × suffisance + 0,15 × profondeur cognitive)",
        "avertissement": (
            "Maîtrise attendue d'un élève suivant ce support, déduite des caractéristiques "
            "du document. Ce n'est pas la mesure de la maîtrise d'un élève réel."
        ),
    }


def _calculer_indicateurs(agent1: dict, agent3: dict, agent4: dict, agent6: dict,
                          difficulte: dict, profils_bloom: dict) -> tuple[dict, dict, dict]:
    # --- Couverture et approfondissement --------------------------------
    couverture = agent6.get("score_global_pct", 0.0) / 100.0
    approfondissement = agent6.get("score_approfondissement_pct", 0.0) / 100.0

    # --- Homogeneite entre referentiels ---------------------------------
    taux = [p["taux_couverture_pct"] for p in (agent6.get("par_pays") or {}).values()]
    homogeneite = (
        float(np.clip(1 - float(np.std(taux)) / 35.0, 0, 1)) if len(taux) > 1 else 0.75
    )

    # --- Volume de contenu par chapitre ----------------------------------
    nb_chapitres = max(1, agent1.get("nb_chapitres", 1))
    profondeur_contenu = float(np.clip(
        (agent1.get("nb_mots", 0) / nb_chapitres) / 250.0, 0, 1
    ))

    # --- Structuration ----------------------------------------------------
    unites = agent3.get("unites", [])
    part_objectifs = (
        sum(1 for u in unites if u.get("objectifs")) / len(unites) if unites else 0.0
    )
    structuration = float(np.clip(
        0.5 * float(np.clip(nb_chapitres / 8.0, 0, 1)) + 0.5 * part_objectifs, 0, 1
    ))

    # --- Richesse pedagogique ---------------------------------------------
    elements = agent1.get("elements_pedagogiques", {}) or {}
    richesse = float(np.clip(sum(elements.values()) / (3.0 * nb_chapitres), 0, 1))

    # --- Diversite lexicale ------------------------------------------------
    mots_cles = agent1.get("mots_cles", [])
    nb_mots = max(1, agent1.get("nb_mots", 1))
    occurrences_top = sum(m["occurrences"] for m in mots_cles[:10])
    diversite = float(np.clip(1 - (occurrences_top / nb_mots) * 2.2, 0, 1))

    # --- Coherence thematique (non supervise) ------------------------------
    coherence, detail_clustering = _coherence_thematique(agent4)

    # --- Profondeur cognitive globale --------------------------------------
    # Moyenne des profondeurs de Bloom chapitre par chapitre : un support dont
    # un seul chapitre monte a « Analyser » n'est pas globalement exigeant.
    indices = [p.get("profondeur_cognitive", 0.0) for p in profils_bloom.values() if p]
    profondeur_cognitive = float(np.mean(indices)) if indices else 0.0

    valeurs = {
        "couverture_internationale": round(couverture, 3),
        "approfondissement": round(approfondissement, 3),
        "homogeneite_couverture": round(homogeneite, 3),
        "profondeur_contenu": round(profondeur_contenu, 3),
        "structuration": round(structuration, 3),
        "richesse_pedagogique": round(richesse, 3),
        "diversite_lexicale": round(diversite, 3),
        "coherence_thematique": round(coherence, 3),
        "adequation_niveau": round(difficulte["adequation"], 3),
        "progression_difficulte": round(difficulte["progression"]["score"], 3),
        "profondeur_cognitive": round(profondeur_cognitive, 3),
    }

    bloom_global = {
        "profondeur_moyenne": round(profondeur_cognitive, 3),
        "niveau_max_atteint": max(
            (p.get("niveau_max") for p in profils_bloom.values() if p.get("niveau_max")),
            key=lambda n: pedagogie.NIVEAUX_BLOOM.index(n),
            default=None,
        ),
        "par_chapitre": {
            titre: {
                "niveau_max": p.get("niveau_max"),
                "profondeur": p.get("profondeur_cognitive"),
                "distribution": p.get("distribution", {}),
            }
            for titre, p in profils_bloom.items()
        },
        "source": next((p.get("source") for p in profils_bloom.values() if p), "aucune"),
    }
    return valeurs, detail_clustering, bloom_global


def _niveau_maturite(note: float) -> tuple[str, str]:
    if note >= 80:
        return "Aligné", "Le support soutient la comparaison avec les référentiels internationaux."
    if note >= 65:
        return "Solide", "Le support est de bonne facture ; quelques écarts ciblés restent à combler."
    if note >= 50:
        return "À consolider", "Les fondamentaux sont présents mais plusieurs notions attendues manquent."
    if note >= 35:
        return "À renforcer", "Des écarts significatifs séparent ce support des référentiels comparés."
    return "À refondre", "Le support s'écarte fortement des attendus internationaux du niveau visé."


# ---------------------------------------------------------------------------
# 3. Appreciation qualitative
# ---------------------------------------------------------------------------

PROMPT = """Tu es un evaluateur pedagogique intervenant dans un processus d'accreditation.

Cours evalue : {titre_cours} ({matiere}, niveau « {niveau} »)

Indicateurs MESURES automatiquement sur le document (echelle 0 a 1) :
{indicateurs}

Analyse de la difficulte du texte :
- difficulte mediane mesuree : {difficulte_mediane} (bande attendue pour ce cycle : {bande})
- regularite de la progression : {progression} (tau de Kendall {tau})
- profondeur cognitive maximale atteinte : {bloom_max}

Note globale calculee par le modele : {note}/100 — niveau « {maturite} »
Taux de couverture par referentiel : {couverture}
Notions attendues non couvertes : {nb_manquantes}
Notions abordees mais traitees trop brievement : {nb_superficielles}

Redige une appreciation d'evaluateur qui s'appuie sur ces mesures, en
distinguant bien ce qui releve du CONTENU (notions presentes ou absentes) de
ce qui releve de la QUALITE DU TRAITEMENT (profondeur, progression, exigence
cognitive).

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour :
{{
  "appreciation_globale": "4 a 5 phrases d'appreciation argumentee",
  "criteres": [
    {{"critere": "nom du critere", "constat": "constat en une phrase", "niveau": "fort" | "moyen" | "faible"}}
  ],
  "risques_accreditation": ["risque 1", "risque 2"],
  "atouts_a_valoriser": ["atout 1", "atout 2"],
  "levier_prioritaire": "la seule chose a corriger en priorite, en une phrase"
}}

Contraintes : francais, 4 a 6 criteres, maximum 4 elements par liste.
"""


def _appreciation_llm(agent2: dict, valeurs: dict, note: float, maturite: str,
                      agent6: dict, difficulte: dict, bloom_global: dict,
                      matiere: str, niveau: str) -> dict:
    indicateurs_txt = "\n".join(
        f"- {libelle} : {valeurs[cle]}" for cle, libelle, _ in INDICATEURS
    )
    couverture_txt = ", ".join(
        f"{p['pays']} {p['taux_couverture_pct']}%"
        for p in (agent6.get("par_pays") or {}).values()
    ) or "(non calculé)"

    prompt = PROMPT.format(
        titre_cours=agent2.get("titre_cours", ""),
        matiere=matiere,
        niveau=niveau,
        indicateurs=indicateurs_txt,
        difficulte_mediane=difficulte.get("mediane"),
        bande=difficulte.get("bande_attendue"),
        progression=difficulte["progression"]["score"],
        tau=difficulte["progression"]["tau_kendall"],
        bloom_max=bloom_global.get("niveau_max_atteint"),
        note=round(note, 1),
        maturite=maturite,
        couverture=couverture_txt,
        nb_manquantes=agent6.get("nb_notions_manquantes", 0),
        nb_superficielles=agent6.get("nb_notions_superficielles", 0),
    )
    try:
        donnees = gemini_client.generate_json(prompt, agent="agent7_evaluation")
        if not isinstance(donnees, dict):
            raise ValueError("structure JSON inattendue")
        donnees["disponible"] = True
        donnees["erreur"] = None
        return donnees
    except Exception as exc:
        criteres = [
            {
                "critere": libelle,
                "constat": f"Indicateur mesuré à {valeurs[cle]} sur 1.",
                "niveau": "fort" if valeurs[cle] >= 0.66 else ("moyen" if valeurs[cle] >= 0.4 else "faible"),
            }
            for cle, libelle, _ in INDICATEURS
        ]
        return {
            "disponible": False,
            "erreur": str(exc),
            "appreciation_globale": (
                f"Le support obtient une note globale de {round(note, 1)}/100 "
                f"(niveau « {maturite} »), calculée par l'ensemble de régression à partir "
                f"des {len(INDICATEURS)} indicateurs mesurés. L'appréciation rédigée n'a pas "
                f"pu être générée (modèle de langage indisponible) ; les indicateurs chiffrés "
                f"restent intégralement exploitables."
            ),
            "criteres": criteres,
            "risques_accreditation": [
                f"{agent6.get('nb_notions_manquantes', 0)} notion(s) attendue(s) non couverte(s).",
                f"{agent6.get('nb_notions_superficielles', 0)} notion(s) traitée(s) trop brièvement.",
            ],
            "atouts_a_valoriser": [c["critere"] for c in criteres if c["niveau"] == "fort"][:4],
            "levier_prioritaire": min(
                INDICATEURS, key=lambda i: valeurs[i[0]] * i[2]
            )[1],
        }


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def process(agent1: dict, agent2: dict, agent3: dict, agent4: dict, agent6: dict,
            matiere: str, niveau: str) -> dict:
    """Execute l'Agent 7."""
    profils_bloom = _bloom_par_chapitre(agent1, agent2)
    difficulte = _analyse_difficulte(agent3, agent4, niveau)

    valeurs, detail_clustering, bloom_global = _calculer_indicateurs(
        agent1, agent3, agent4, agent6, difficulte, profils_bloom
    )

    modele = _obtenir_modele()
    vecteur = np.array([[valeurs[cle] for cle, _, _ in INDICATEURS]])
    note = float(np.clip(modele.predict(vecteur)[0], 0, 100))

    # Divergence entre les deux tetes : indicateur d'incertitude du modele.
    note_gbr = float(np.clip(modele.named_estimators_["gbr"].predict(vecteur)[0], 0, 100))
    note_mlp = float(np.clip(modele.named_estimators_["mlp"].predict(vecteur)[0], 0, 100))
    divergence = abs(note_gbr - note_mlp)

    maturite, message_maturite = _niveau_maturite(note)

    contributions = [
        {
            "cle": cle,
            "libelle": libelle,
            "valeur": valeurs[cle],
            "poids_pct": round(poids * 100, 1),
            "contribution_points": round(valeurs[cle] * poids * 100, 1),
            "niveau": "fort" if valeurs[cle] >= 0.66 else ("moyen" if valeurs[cle] >= 0.4 else "faible"),
        }
        for cle, libelle, poids in INDICATEURS
    ]

    profil_maitrise = _profil_maitrise(agent6, profils_bloom)

    appreciation = _appreciation_llm(
        agent2, valeurs, note, maturite, agent6, difficulte, bloom_global, matiere, niveau
    )

    return {
        "indicateurs": valeurs,
        "contributions": contributions,
        "note_globale": round(note, 1),
        "notes_par_tete": {
            "gradient_boosting": round(note_gbr, 1),
            "reseau_de_neurones": round(note_mlp, 1),
            "divergence": round(divergence, 1),
            "fiabilite": "élevée" if divergence < 4 else ("moyenne" if divergence < 9 else "faible"),
        },
        "niveau_maturite": maturite,
        "message_maturite": message_maturite,
        "clustering": detail_clustering,
        "difficulte": difficulte,
        "bloom": bloom_global,
        "profil_maitrise": profil_maitrise,
        "modele": infos_modele(),
        "appreciation": appreciation,
    }
