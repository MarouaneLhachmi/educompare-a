"""
Outils d'analyse pedagogique partages.
=======================================

Ce module regroupe deux mesures pedagogiques mobilisees par plusieurs agents,
afin qu'elles soient definies une seule fois et restent coherentes entre eux :

- la **taxonomie de Bloom** : a quel niveau cognitif un objectif ou un
  exercice sollicite-t-il l'eleve ? L'Agent 7 s'en sert pour evaluer la
  profondeur cognitive du support, l'Agent 8 pour cibler le niveau vise par
  chaque intervention qu'il recommande ;

- la **difficulte linguistique** d'un texte, estimee par une formule de
  lisibilite. L'Agent 7 en deduit le niveau reel du support et la regularite
  de sa progression.

Chaque mesure dispose d'une implementation deterministe immediatement
operationnelle, et bascule automatiquement sur un modele entraine des que
l'artefact correspondant est depose dans `app/models/`.
"""

import re
import unicodedata

import numpy as np

from app.services import model_registry

# ---------------------------------------------------------------------------
# Taxonomie de Bloom
# ---------------------------------------------------------------------------
# Les six niveaux, du plus elementaire au plus exigeant (revision Anderson &
# Krathwohl, 2001). L'indice sert d'echelle ordinale dans les calculs.
NIVEAUX_BLOOM = [
    "Mémoriser",
    "Comprendre",
    "Appliquer",
    "Analyser",
    "Évaluer",
    "Créer",
]

# Verbes d'action caracteristiques de chaque niveau, en francais et en anglais
# (les referentiels etrangers indexes sont partiellement anglophones).
VERBES_BLOOM = {
    "Mémoriser": [
        "definir", "nommer", "citer", "reciter", "identifier", "reconnaitre",
        "lister", "rappeler", "enumerer", "memoriser", "connaitre", "situer",
        "lire", "reperer",
        "define", "name", "list", "recall", "identify", "recognise", "recognize",
        "state", "label", "read",
    ],
    "Comprendre": [
        "expliquer", "decrire", "resumer", "illustrer", "interpreter",
        "reformuler", "classer", "distinguer", "comprendre", "traduire",
        "comparer", "ordonner", "ranger", "trier",
        "explain", "describe", "summarise", "summarize", "interpret",
        "classify", "understand", "compare", "order", "sort",
    ],
    "Appliquer": [
        "calculer", "resoudre", "utiliser", "appliquer", "construire",
        "tracer", "mesurer", "convertir", "executer", "effectuer", "realiser",
        "employer", "manipuler", "poser", "ecrire", "completer", "reproduire",
        "additionner", "soustraire", "multiplier", "diviser",
        "calculate", "solve", "use", "apply", "construct", "draw", "measure",
        "convert", "perform", "multiply", "divide", "add", "subtract", "write",
    ],
    "Analyser": [
        "analyser", "decomposer", "differencier", "organiser", "structurer",
        "deduire", "examiner", "categoriser", "relier", "confronter",
        "analyse", "analyze", "differentiate", "organise", "organize",
        "deduce", "examine", "investigate", "relate",
    ],
    "Évaluer": [
        "evaluer", "justifier", "critiquer", "argumenter", "verifier",
        "valider", "juger", "defendre", "apprecier", "controler", "estimer",
        "evaluate", "justify", "argue", "verify", "validate", "judge",
        "assess", "check", "estimate",
    ],
    "Créer": [
        "creer", "concevoir", "inventer", "elaborer", "produire", "formuler",
        "composer", "planifier", "modeliser", "imaginer", "generer",
        "create", "design", "invent", "produce", "formulate", "compose",
        "plan", "model", "generate", "devise",
    ],
}

_INDEX_BLOOM = {niveau: i for i, niveau in enumerate(NIVEAUX_BLOOM)}


def _sans_accents(texte: str) -> str:
    normalise = unicodedata.normalize("NFKD", (texte or "").lower())
    return "".join(c for c in normalise if not unicodedata.combining(c))


def classer_bloom(texte: str) -> dict:
    """
    Situe un enonce sur la taxonomie de Bloom.

    Repli deterministe : reperage des verbes d'action caracteristiques. Le
    niveau retenu est le **plus eleve** effectivement detecte — un exercice
    qui demande de calculer puis de justifier releve bien de « Évaluer ».
    Ce choix est volontairement optimiste : il evite de sous-estimer un
    support riche a cause d'une formulation majoritairement descriptive.
    """
    modele = model_registry.charger("bloom_clf")
    if modele is not None:
        try:
            prediction = modele.predict([texte])[0]
            niveau = NIVEAUX_BLOOM[int(prediction)] if isinstance(prediction, (int, np.integer)) else str(prediction)
            return {
                "niveau": niveau,
                "indice": _INDEX_BLOOM.get(niveau, 0),
                "source": "modele_entraine",
                "verbes_detectes": [],
            }
        except Exception:
            pass

    mots = set(re.findall(r"[a-z]{3,}", _sans_accents(texte)))

    # Appariement volontairement strict : infinitif exact, ou gerondif en
    # « -ant » ramene a son infinitif. Un appariement par prefixe serait
    # tentant mais produit des faux positifs graves — « la formule » (nom
    # tres frequent en mathematiques) declencherait le verbe « formuler » et
    # ferait basculer un exercice de calcul au niveau « Créer ». Dans des
    # objectifs pedagogiques, les verbes sont de toute facon presque toujours
    # a l'infinitif : la precision prime ici sur le rappel.
    candidats = set(mots)
    for mot in mots:
        if mot.endswith("ant") and len(mot) > 5:
            candidats.add(mot[:-3] + "er")

    detectes: list[tuple[int, str]] = []
    for niveau, verbes in VERBES_BLOOM.items():
        for verbe in verbes:
            if _sans_accents(verbe) in candidats:
                detectes.append((_INDEX_BLOOM[niveau], verbe))
                break

    if not detectes:
        return {
            "niveau": NIVEAUX_BLOOM[1],  # « Comprendre » par defaut
            "indice": 1,
            "source": "defaut",
            "verbes_detectes": [],
        }

    indice = max(i for i, _ in detectes)
    return {
        "niveau": NIVEAUX_BLOOM[indice],
        "indice": indice,
        "source": "heuristique_verbale",
        "verbes_detectes": [v for i, v in detectes if i == indice],
    }


def profil_bloom(textes: list[str]) -> dict:
    """
    Profil cognitif d'un ensemble d'enonces : distribution sur les six
    niveaux, niveau maximal atteint et **profondeur cognitive** normalisee.

    La profondeur est la moyenne des indices ramenee sur [0, 1]. Un support
    qui se limite a « memoriser / comprendre » plafonne autour de 0,2 ; un
    support qui fait analyser et creer depasse 0,6.
    """
    textes = [t for t in textes if t and t.strip()]
    if not textes:
        return {
            "distribution": {n: 0 for n in NIVEAUX_BLOOM},
            "niveau_max": None,
            "indice_max": 0,
            "profondeur_cognitive": 0.0,
            "source": "aucun_enonce",
        }

    classements = [classer_bloom(t) for t in textes]
    distribution = {n: 0 for n in NIVEAUX_BLOOM}
    for c in classements:
        distribution[c["niveau"]] += 1

    indices = [c["indice"] for c in classements]
    indice_max = max(indices)
    return {
        "distribution": distribution,
        "distribution_pct": {
            n: round(100 * v / len(classements), 1) for n, v in distribution.items()
        },
        "niveau_max": NIVEAUX_BLOOM[indice_max],
        "indice_max": indice_max,
        "profondeur_cognitive": round(float(np.mean(indices)) / (len(NIVEAUX_BLOOM) - 1), 3),
        "source": classements[0]["source"],
    }


# ---------------------------------------------------------------------------
# Difficulte linguistique
# ---------------------------------------------------------------------------

_VOYELLES = re.compile(r"[aeiouyàâäéèêëîïôöùûü]+", re.IGNORECASE)
_PHRASES = re.compile(r"[.!?]+")
_MOTS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def _compter_syllabes(mot: str) -> int:
    """Approximation par groupes de voyelles — suffisante pour un indice
    agrege sur plusieurs centaines de mots."""
    return max(1, len(_VOYELLES.findall(mot)))


def indice_lisibilite(texte: str) -> float:
    """
    Indice de lisibilite de Flesch adapte au francais par Kandel et Moles :

        207 − 1,015 × (mots / phrases) − 73,6 × (syllabes / mots)

    Resultat sur une echelle de 0 (tres difficile) a 100 (tres facile).
    """
    mots = _MOTS.findall(texte or "")
    if not mots:
        return 50.0
    phrases = [p for p in _PHRASES.split(texte) if p.strip()]
    nb_phrases = max(1, len(phrases))
    syllabes = sum(_compter_syllabes(m) for m in mots)

    score = 207 - 1.015 * (len(mots) / nb_phrases) - 73.6 * (syllabes / len(mots))
    return float(np.clip(score, 0.0, 100.0))


def difficulte_texte(texte: str) -> float:
    """
    Difficulte estimee d'un texte, sur [0, 1].

    Combine la lisibilite (formule de Kandel-Moles) et deux marqueurs de
    technicite : la proportion de mots longs et la densite de notation
    numerique ou symbolique, tous deux caracteristiques d'un contenu
    scientifique plus exigeant.
    """
    mots = _MOTS.findall(texte or "")
    if not mots:
        return 0.5

    lisibilite = indice_lisibilite(texte) / 100.0
    part_mots_longs = sum(1 for m in mots if len(m) >= 9) / len(mots)
    densite_symboles = len(re.findall(r"[0-9=+×÷%<>/·^]", texte or "")) / max(1, len(texte or ""))

    difficulte = (
        0.60 * (1.0 - lisibilite)
        + 0.25 * min(1.0, part_mots_longs * 4.0)
        + 0.15 * min(1.0, densite_symboles * 25.0)
    )
    return float(np.clip(difficulte, 0.0, 1.0))


def estimer_niveaux(textes: list[str], embeddings=None) -> dict:
    """
    Estime la difficulte de chaque unite de contenu.

    Si l'artefact `niveau_reg` (regression entrainee sur des embeddings, issue
    du notebook 02) est disponible et que les embeddings sont fournis, il est
    utilise ; sinon on retombe sur l'indice de lisibilite ci-dessus.
    """
    if not textes:
        return {"difficultes": [], "source": "aucun_contenu"}

    modele = model_registry.charger("niveau_reg")
    if modele is not None and embeddings is not None and len(embeddings) == len(textes):
        try:
            predictions = np.asarray(modele.predict(embeddings), dtype="float64")
            # Le modele predit un niveau scolaire ; on le ramene sur [0, 1]
            # en supposant une echelle de 1 a 12 annees de scolarite.
            difficultes = np.clip((predictions - 1.0) / 11.0, 0.0, 1.0)
            return {
                "difficultes": [round(float(d), 3) for d in difficultes],
                "niveaux_scolaires": [round(float(p), 2) for p in predictions],
                "source": "modele_entraine",
            }
        except Exception:
            pass

    return {
        "difficultes": [round(difficulte_texte(t), 3) for t in textes],
        "niveaux_scolaires": None,
        "source": "lisibilite_kandel_moles",
    }


def progression_difficulte(difficultes: list[float]) -> dict:
    """
    Regularite de la montee en difficulte au fil du support.

    Un cours bien construit progresse : les premieres unites sont plus
    accessibles que les dernieres. On mesure donc :

    - la **pente** d'une regression lineaire de la difficulte sur le rang ;
    - le **tau de Kendall**, qui capte la monotonie sans supposer de relation
      lineaire — c'est la mesure la plus robuste sur un petit nombre d'unites ;
    - les **ruptures**, ces sauts de difficulte trop brutaux d'une unite a la
      suivante, qui signalent une marche trop haute pour l'eleve.

    Le score retourne vaut 1 pour une progression reguliere et croissante,
    0,5 pour une difficulte plate, et se degrade si le support redescend ou
    comporte des ruptures.
    """
    n = len(difficultes)
    if n < 3:
        return {
            "score": 0.5,
            "pente": 0.0,
            "tau_kendall": 0.0,
            "ruptures": [],
            "mesurable": False,
        }

    y = np.asarray(difficultes, dtype="float64")
    x = np.arange(n, dtype="float64")
    pente = float(np.polyfit(x, y, 1)[0])

    # Tau de Kendall calcule directement : le nombre d'unites reste petit.
    concordants = discordants = 0
    for i in range(n):
        for j in range(i + 1, n):
            ecart = y[j] - y[i]
            if ecart > 1e-9:
                concordants += 1
            elif ecart < -1e-9:
                discordants += 1
    paires = concordants + discordants
    tau = (concordants - discordants) / paires if paires else 0.0

    ecarts = np.abs(np.diff(y))
    seuil_rupture = max(0.25, float(np.mean(ecarts) + 2 * np.std(ecarts)))
    ruptures = [
        {"entre": [i, i + 1], "ecart": round(float(ecarts[i]), 3)}
        for i in range(len(ecarts))
        if ecarts[i] > seuil_rupture
    ]

    # tau ∈ [-1, 1] ramene sur [0, 1], puis penalite de rupture.
    score = (tau + 1.0) / 2.0
    score -= 0.08 * len(ruptures)
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "pente": round(pente, 4),
        "tau_kendall": round(float(tau), 3),
        "ruptures": ruptures,
        "mesurable": True,
    }
