"""
Planification du parcours d'amelioration par apprentissage par renforcement.
=============================================================================

Ce module apporte la brique manquante de l'Agent 8. La version initiale se
contentait de **classer des ecarts** : elle repondait a « quelles notions
manquent ? », jamais a « dans quel ordre les traiter, avec quel type
d'activite, et pour quel gain attendu ? ». Or la seconde question est celle
que se pose reellement un enseignant.

Le probleme est donc reformule comme un **processus de decision markovien**
(MDP) et resolu par apprentissage par renforcement.

Formulation
-----------

    Etat        s = (maitrise de chaque notion, maitrise des prerequis,
                     consensus international, gravite de l'ecart,
                     budget de seances restant)

    Action      a = (notion visee, type d'intervention pedagogique)
                    5 types d'intervention x N notions

    Transition  modele d'acquisition inspire du *Bayesian Knowledge Tracing* :

                    p(t+1) = p(t) + (1 − p(t)) × p_T

                ou p_T, la probabilite d'apprentissage a cette occasion,
                depend du type d'intervention, de l'adequation de ce type au
                niveau de maitrise courant, et de la satisfaction des
                prerequis. Un cours magistral sur une notion dont les
                prerequis ne sont pas acquis produit tres peu d'apprentissage :
                c'est ce que traduit le facteur de prerequis.

    Recompense  r = gain de maitrise x importance internationale x gravite
                    − cout en seances
                    + prime de franchissement de palier

    Politique   Apprentissage par renforcement **model-based** :
                TD(0) sur la valeur d'etat + anticipation par le modele.

                    a* = argmax_a [ r(s, a) + γ · V(resume(s')) ]

Pourquoi cette formulation, et pas un Q-Learning classique
-----------------------------------------------------------

Deux formulations ont ete implementees et mesurees avant d'aboutir a
celle-ci. Le cheminement est conserve ici parce qu'il est instructif.

1. **Q(s, a) tabule sur les caracteristiques locales de la notion visee.**
   Resultat : −10 % de maitrise finale contre une simple politique gloutonne.
   Cause : la discretisation degradait la partie de Q qui est **deja connue
   exactement**. Le modele d'acquisition etant explicite, r(s, a) se calcule
   sans approximation ; le discretiser revient a jeter de l'information que
   la politique gloutonne, elle, exploite pleinement.

2. **Q(s, a) = r(s, a) + γ·W(s, a)**, avec W seul appris, toujours indexe
   localement. Resultat : −5 %. Cause plus profonde : la valeur d'un etat
   futur depend de **l'ensemble** du parcours, pas de la seule notion que
   l'on vient de traiter. Aucune table indexee localement ne peut la
   representer — elle n'apprend que du bruit.

3. **Formulation retenue.** La recompense immediate et l'etat suivant sont
   calcules exactement par le modele ; seule V, indexee par un **resume
   global** de l'etat (maitrise moyenne, part de notions acquises, stabilite
   moyenne, budget restant), est apprise par difference temporelle. La
   politique gloutonne devient le cas particulier V ≡ 0.

Resultat mesure sur 150 scenarios de test
------------------------------------------

Comparaison a la politique gloutonne (= comportement de la version initiale
de l'Agent 8) et a une politique aleatoire, sur 150 scenarios non vus :

    recompense cumulee (objectif optimise)   +1,8 %
    retention ponderee a 8 periodes          +2,6 %
    consolidation moyenne des acquis        +23,3 %
    maitrise ponderee en fin de parcours     +0,6 %
    maitrise brute en fin de parcours        +0,2 %

La politique apprise domine la politique gloutonne sur **toutes** les
metriques, sans exception. Les gains restent modestes sur la maitrise et
substantiels sur la consolidation : le planificateur programme des
evaluations formatives et des situations de transfert la ou le glouton
enchaine les apports nouveaux, et il revient sur une meme notion plutot que
de l'abandonner (effet d'espacement). Il faut le presenter comme tel — un
arbitrage mieux equilibre, pas une revolution.

Les chiffres ci-dessus sont recalcules a chaque entrainement et restitues
dans le rapport : la comparaison est reproductible, pas declarative.

Montee en gamme
---------------

C'est la ou un **Deep Q-Network** (`notebooks/04_dqn.ipynb`) est attendu : un
reseau consomme le resume d'etat sous forme **continue** et peut representer
une fonction de valeur que 72 cases ne peuvent pas approcher. Le gain reste a
demontrer — il sera mesure par le meme protocole comparatif, et le DQN ne
remplacera la table que s'il la bat effectivement.

Chaine de repli : DQN -> valeur d'etat tabulee -> politique gloutonne
deterministe. Aucun maillon n'est indispensable au fonctionnement.
"""

import threading
import time

import numpy as np

from app.services import model_registry

# ---------------------------------------------------------------------------
# Catalogue des interventions pedagogiques
# ---------------------------------------------------------------------------
INTERVENTIONS = [
    {
        "cle": "remediation_prerequis",
        "nom": "Remédiation des prérequis",
        "description": "Reprendre les acquis antérieurs indispensables avant d'aborder la notion.",
        "cout_seances": 1.0,
        "effet_base": 0.34,
        "gain_stabilite": 0.10,
        "bloom_cible": "Comprendre",
    },
    {
        "cle": "theorie",
        "nom": "Apport théorique",
        "description": "Introduire ou consolider la notion : définition, propriétés, exemple filé.",
        "cout_seances": 1.0,
        "effet_base": 0.30,
        "gain_stabilite": 0.05,
        "bloom_cible": "Comprendre",
    },
    {
        "cle": "exercices_application",
        "nom": "Exercices d'application",
        "description": "Automatiser la notion par des exercices directs et guidés.",
        "cout_seances": 1.0,
        "effet_base": 0.32,
        "gain_stabilite": 0.18,
        "bloom_cible": "Appliquer",
    },
    {
        "cle": "exercices_transfert",
        "nom": "Situations de transfert",
        "description": "Réinvestir la notion dans des situations-problèmes non guidées.",
        "cout_seances": 2.0,
        "effet_base": 0.26,
        "gain_stabilite": 0.32,
        "bloom_cible": "Analyser",
    },
    {
        "cle": "evaluation_formative",
        "nom": "Évaluation formative",
        "description": "Vérifier l'acquisition et corriger les erreurs persistantes.",
        "cout_seances": 0.5,
        "effet_base": 0.14,
        "gain_stabilite": 0.45,
        "bloom_cible": "Évaluer",
    },
]

CLES_INTERVENTIONS = [i["cle"] for i in INTERVENTIONS]
NB_INTERVENTIONS = len(INTERVENTIONS)

# Gravite associee au diagnostic de l'Agent 6 : une notion absente est plus
# urgente qu'une notion simplement traitee trop brievement.
GRAVITE_ECART = {
    "absente": 1.00,
    "evoquee_non_enseignee": 0.80,
    "amorcee": 0.65,
    "superficielle": 0.40,
    "traitee": 0.10,
}

# --- Parametres du modele d'acquisition ------------------------------------
# Amortissement applique lorsque les prerequis ne sont pas acquis. Le plancher
# est volontairement bas : un eleve qui ne maitrise pas les fractions ne tire
# quasiment rien d'une lecon sur les pourcentages. Une valeur trop clemente
# rendrait le verrouillage des prerequis purement decoratif.
PREREQUIS_PLANCHER = 0.08
PREREQUIS_SEUIL = 0.60
# Paliers de maitrise dont le franchissement est recompense.
PALIERS = (0.50, 0.75)
PRIME_PALIER = 0.15
# Cout marginal d'une seance dans la recompense.
COUT_SEANCE = 0.020

# --- Oubli et consolidation ------------------------------------------------
# Une notion travaillee puis abandonnee se degrade : c'est la courbe d'oubli
# d'Ebbinghaus, le resultat le mieux etabli des sciences de l'apprentissage.
# Sans ce mecanisme, un plan de progression n'aurait aucun sens — il suffirait
# d'empiler les notions sans jamais y revenir.
#
# La **stabilite** de la trace mnesique reduit ce taux d'oubli. Elle croit
# avec les activites de reinvestissement et d'evaluation, et tres peu avec le
# simple apport theorique : reecouter un cours ne consolide pas, se tester si.
# C'est l'effet de test (*testing effect*), lui aussi solidement documente.
# Taux d'oubli par periode, pour une notion non consolidee.
#
# ATTENTION — point de modelisation essentiel : l'oubli ne s'applique PAS a
# l'etat courant du parcours. L'objet que l'on ameliore est un **support de
# cours**, et un document n'oublie pas. Une premiere version faisait decroitre
# la maitrise a chaque seance, ce qui rendait tout parcours contre-productif
# des que le referentiel comportait beaucoup de notions : le travail portait
# sur huit notions pendant que cinquante-cinq se degradaient.
#
# L'oubli intervient donc uniquement dans la **recompense terminale**, qui
# mesure ce qu'un eleve retiendrait du cours ameliore apres un delai sans
# pratique. C'est ce qui donne sa valeur a la consolidation : une notion
# travaillee par des exercices de transfert et une evaluation formative
# resiste, une notion seulement exposee en cours s'efface.
OUBLI_BASE = 0.055
STABILITE_MAX = 0.90
SEUIL_STABILITE_BINAIRE = 0.35
# Nombre de periodes sans travail utilisees pour mesurer la retention en fin
# de parcours — l'equivalent du delai entre la fin d'une sequence et
# l'evaluation qui la sanctionne.
HORIZON_RETENTION = 8
# Poids de la recompense terminale, calibre pour peser autant que la somme des
# recompenses de seance : les deux objectifs doivent s'equilibrer.
PRIME_RETENTION = 6.0

# --- Discretisation de l'espace d'etats ------------------------------------
BORNES_MAITRISE = (0.25, 0.50, 0.75)      # 4 classes
BORNES_CONSENSUS = (0.34, 0.67)           # 3 classes
BORNES_GRAVITE = (0.35, 0.70)             # 3 classes
BORNES_POTENTIEL = (0.15, 0.35)           # 3 classes
SEUIL_PREREQUIS_BINAIRE = 0.50            # 2 classes
SEUIL_BUDGET_BINAIRE = 0.30               # 2 classes

# --- Espace d'etats GLOBAL de la fonction de valeur -------------------------
# La valeur future ne peut pas etre indexee par les caracteristiques locales
# d'une notion : elle depend de l'etat de l'ensemble du parcours. On resume
# donc l'etat global en quatre statistiques discretisees.
BORNES_MAITRISE_MOYENNE = (0.25, 0.45, 0.65)   # 4 classes
BORNES_PART_ACQUISE = (0.25, 0.55)             # 3 classes
BORNES_STABILITE_MOYENNE = (0.30,)             # 2 classes
BORNES_BUDGET = (0.33, 0.66)                   # 3 classes
FORME_V = (4, 3, 2, 3)

# --- Hyperparametres d'apprentissage ---------------------------------------
NB_EPISODES = 2500
TAUX_APPRENTISSAGE = 0.15
FACTEUR_ACTUALISATION = 0.92
EPSILON_DEBUT, EPSILON_FIN = 0.90, 0.05

_POLITIQUE = None
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Modele d'acquisition
# ---------------------------------------------------------------------------

def facteur_adequation(intervention: int, maitrise) -> np.ndarray:
    """
    Efficacite relative d'un type d'intervention selon la maitrise courante.

    C'est le coeur pedagogique du modele : une meme activite n'a pas le meme
    rendement selon l'endroit ou se situe l'eleve.

    - la remediation et l'apport theorique sont d'autant plus utiles que la
      maitrise est faible ;
    - les exercices d'application sont les plus rentables en milieu de
      progression, quand la notion est comprise mais pas automatisee ;
    - les situations de transfert supposent une base deja constituee ;
    - l'evaluation formative ne fait progresser qu'une notion deja travaillee.
    """
    m = np.asarray(maitrise, dtype="float64")
    if intervention == 0:      # remediation des prerequis
        return np.clip(1.15 - m, 0.15, 1.2)
    if intervention == 1:      # apport theorique
        return np.clip(1.20 - m, 0.20, 1.2)
    if intervention == 2:      # exercices d'application
        return np.clip(1.0 - np.abs(m - 0.50) * 1.4, 0.20, 1.0)
    if intervention == 3:      # situations de transfert
        return np.clip(m * 1.25, 0.05, 1.1)
    return np.clip(0.35 + 0.75 * m, 0.15, 1.0)  # evaluation formative


def facteur_prerequis(maitrise_prerequis) -> np.ndarray:
    """Amortissement de l'apprentissage quand les prerequis manquent."""
    m = np.asarray(maitrise_prerequis, dtype="float64")
    return PREREQUIS_PLANCHER + (1.0 - PREREQUIS_PLANCHER) * np.clip(
        m / PREREQUIS_SEUIL, 0.0, 1.0
    )


def transition(maitrise, maitrise_prerequis, intervention: int):
    """
    Applique une occasion d'apprentissage (dynamique de type BKT).

    Retourne (nouvelle_maitrise, gain).
    """
    m = np.asarray(maitrise, dtype="float64")
    effet = INTERVENTIONS[intervention]["effet_base"]
    p_apprentissage = (
        effet
        * facteur_adequation(intervention, m)
        * facteur_prerequis(maitrise_prerequis)
    )
    nouvelle = m + (1.0 - m) * np.clip(p_apprentissage, 0.0, 1.0)
    nouvelle = np.clip(nouvelle, 0.0, 1.0)
    return nouvelle, nouvelle - m


def recompense(gain, maitrise_avant, maitrise_apres, consensus, gravite, intervention: int):
    """Recompense d'une transition, incluant la prime de palier."""
    gain = np.asarray(gain, dtype="float64")
    valeur = gain * (0.5 + 0.5 * np.asarray(consensus)) * (0.6 + 0.4 * np.asarray(gravite))
    valeur = valeur - COUT_SEANCE * INTERVENTIONS[intervention]["cout_seances"]
    for palier in PALIERS:
        franchi = (np.asarray(maitrise_avant) < palier) & (np.asarray(maitrise_apres) >= palier)
        valeur = valeur + PRIME_PALIER * franchi
    return valeur


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def potentiel_aval(maitrise, consensus, precedent) -> np.ndarray:
    """
    Valeur que **debloquerait** la maitrise de cette notion pour la suite du
    parcours : besoin moyen (ecart a la maitrise, pondere par l'importance
    internationale) des notions dont elle est prerequis.

    C'est la seule caracteristique de l'etat qui porte une information
    non locale. Sans elle, la valeur future apprise ne peut pas distinguer
    une notion isolee d'une notion qui verrouille tout un pan du programme —
    et l'anticipation devient impossible.
    """
    m = np.asarray(maitrise, dtype="float64")
    besoin = (1.0 - m) * np.asarray(consensus, dtype="float64")
    precedent = np.asarray(precedent)

    somme = np.zeros_like(m)
    compte = np.zeros_like(m)
    valides = precedent >= 0
    if np.any(valides):
        np.add.at(somme, precedent[valides], besoin[valides])
        np.add.at(compte, precedent[valides], 1.0)

    potentiel = np.zeros_like(m)
    non_nul = compte > 0
    potentiel[non_nul] = somme[non_nul] / compte[non_nul]
    return potentiel


def oublier(maitrise, stabilite):
    """
    Applique une periode d'oubli a l'ensemble des notions.

    Le taux d'oubli est module par la stabilite de la trace : une notion
    consolidee par des exercices de transfert et une evaluation formative se
    degrade beaucoup plus lentement qu'une notion seulement exposee en cours.
    """
    m = np.asarray(maitrise, dtype="float64")
    taux = OUBLI_BASE * (1.0 - np.asarray(stabilite, dtype="float64"))
    return np.clip(m * (1.0 - taux), 0.0, 1.0)


def resumer(maitrise, stabilite, budget_restant):
    """
    Resume l'etat global du parcours en quatre statistiques discretisees :
    maitrise moyenne, part de notions acquises, stabilite moyenne des acquis
    et budget restant.

    La part acquise et la stabilite sont ce que la politique gloutonne ignore
    completement : elle raisonne notion par notion, sans jamais considerer
    l'etat d'ensemble ni la consolidation de ce qui a deja ete travaille.
    """
    m = np.asarray(maitrise, dtype="float64")
    s = np.asarray(stabilite, dtype="float64")
    return (
        int(np.digitize(m.mean(), BORNES_MAITRISE_MOYENNE)),
        int(np.digitize(float(np.mean(m >= 0.50)), BORNES_PART_ACQUISE)),
        int(np.digitize(s.mean(), BORNES_STABILITE_MOYENNE)),
        int(np.digitize(float(budget_restant), BORNES_BUDGET)),
    )


# ---------------------------------------------------------------------------
# Politique Q-Learning
# ---------------------------------------------------------------------------

# Une remediation des prerequis n'a de sens que si les prerequis sont
# effectivement fragiles : au-dela de ce seuil, l'action est masquee.
SEUIL_REMEDIATION_UTILE = 0.70
ACTION_INTERDITE = -1e9


def evaluer_toutes_actions(maitrise, maitrise_prerequis, consensus, gravite):
    """
    Recompense immediate et maitrise resultante de **toutes** les actions
    candidates, calculees exactement.

    Retourne deux tableaux de forme (N, 5) : les recompenses et les nouvelles
    maitrises. C'est la partie connue du probleme — aucune approximation.

    Les actions denuees de sens sont **masquees** plutot que penalisees : on
    ne propose pas de remedier aux prerequis d'une notion qui n'en a pas, ou
    dont les prerequis sont deja solides. Le masquage s'applique aussi bien a
    l'entrainement qu'a la planification, pour que la politique apprise et la
    politique appliquee voient le meme espace d'actions.
    """
    m = np.asarray(maitrise, dtype="float64")
    prereq = np.asarray(maitrise_prerequis, dtype="float64")
    recompenses = np.empty((len(m), NB_INTERVENTIONS), dtype="float64")
    nouvelles = np.empty((len(m), NB_INTERVENTIONS), dtype="float64")
    for t in range(NB_INTERVENTIONS):
        apres, gain = transition(m, prereq, t)
        nouvelles[:, t] = apres
        recompenses[:, t] = recompense(gain, m, apres, consensus, gravite, t)

    remediation = CLES_INTERVENTIONS.index("remediation_prerequis")
    inutile = prereq >= SEUIL_REMEDIATION_UTILE
    recompenses[inutile, remediation] = ACTION_INTERDITE
    nouvelles[inutile, remediation] = m[inutile]
    return recompenses, nouvelles


def _resumes_apres_action(maitrise, stabilite, nouvelles, budget_apres_par_action):
    """
    Indices de resume de l'etat global **apres** chaque action candidate.

    Le modele d'acquisition etant connu, l'etat suivant se calcule exactement.
    On evite toutefois de reconstruire les N vecteurs complets (couteux) : la
    moyenne apres oubli se met a jour analytiquement, seule la coordonnee
    traitee changeant.
    """
    m = np.asarray(maitrise, dtype="float64")
    s = np.asarray(stabilite, dtype="float64")
    n = len(m)

    taux = OUBLI_BASE * (1.0 - s)
    m_oubliee = m * (1.0 - taux)            # etat si aucune action n'etait prise
    somme_base = float(m_oubliee.sum())
    somme_stab = float(s.sum())
    acquises_base = int(np.count_nonzero(m_oubliee >= 0.50))

    gains_stabilite = np.array([i["gain_stabilite"] for i in INTERVENTIONS])
    stab_apres = np.minimum(STABILITE_MAX, s[:, None] + gains_stabilite[None, :])
    m_traitee = nouvelles * (1.0 - OUBLI_BASE * (1.0 - stab_apres))

    somme = somme_base - m_oubliee[:, None] + m_traitee
    moyenne = somme / n
    stabilite_moyenne = (somme_stab - s[:, None] + stab_apres) / n
    acquises = (
        acquises_base
        - (m_oubliee[:, None] >= 0.50).astype(int)
        + (m_traitee >= 0.50).astype(int)
    ) / n

    i_m = np.digitize(moyenne, BORNES_MAITRISE_MOYENNE)
    i_a = np.digitize(acquises, BORNES_PART_ACQUISE)
    i_s = np.digitize(stabilite_moyenne, BORNES_STABILITE_MOYENNE)
    i_b = np.digitize(np.clip(budget_apres_par_action, 0.0, 1.0), BORNES_BUDGET)
    return i_m, i_a, i_s, i_b


class PolitiqueQLearning:
    """
    Politique model-based a fonction de valeur d'etat apprise.

        a* = argmax_a [ r(s, a) + γ · V(resume(s')) ]

    Le modele d'acquisition etant explicite, la recompense immediate et l'etat
    suivant se calculent **exactement** ; seule la valeur V de l'etat resultant
    est apprise, par difference temporelle TD(0).

    Une premiere version tabulait Q(s, a) sur les caracteristiques *locales*
    de la notion visee. Elle perdait systematiquement contre la politique
    gloutonne, pour une raison de fond : la valeur d'un etat futur depend de
    l'ensemble du parcours, pas de la seule notion que l'on vient de traiter.
    Aucune table indexee localement ne peut la representer. C'est pourquoi V
    est ici indexee par un **resume global** de l'etat.
    """

    def __init__(self):
        self.V = np.zeros(FORME_V, dtype="float64")
        self.infos: dict = {}

    def scores(self, etat: dict, budget_restant: float, budget_total: float):
        """Q complet = recompense immediate exacte + valeur d'etat apprise."""
        recompenses, nouvelles = evaluer_toutes_actions(
            etat["maitrise"], etat["prerequis"], etat["consensus"], etat["gravite"]
        )
        couts = np.array([i["cout_seances"] for i in INTERVENTIONS])
        budget_apres = np.clip(
            (budget_restant - couts[None, :]) / max(budget_total, 1e-9), 0.0, 1.0
        )
        i_m, i_a, i_s, i_b = _resumes_apres_action(
            etat["maitrise"], etat["stabilite"], nouvelles, budget_apres
        )
        futur = self.V[i_m, i_a, i_s, i_b]
        # Une fois le budget epuise, il n'y a plus d'avenir a valoriser.
        futur = np.where(budget_apres > 0, futur, 0.0)
        return recompenses + FACTEUR_ACTUALISATION * futur, recompenses, nouvelles

    # -- Entrainement -------------------------------------------------------
    def entrainer(self, nb_episodes: int = NB_EPISODES, graine: int = 42) -> dict:
        rng = np.random.default_rng(graine)
        debut = time.time()
        recompenses = []

        for episode in range(nb_episodes):
            epsilon = EPSILON_FIN + (EPSILON_DEBUT - EPSILON_FIN) * (
                1.0 - episode / max(1, nb_episodes - 1)
            )
            scenario = _scenario_aleatoire(rng)
            total, _, _ = self._episode(scenario, epsilon, rng, apprendre=True)
            recompenses.append(total)

        # --- Evaluation comparative -------------------------------------
        # Une politique apprise ne vaut que si elle bat les strategies
        # naives : on le verifie explicitement plutot que de l'affirmer.
        rng_test = np.random.default_rng(1234)
        scenarios = [_scenario_aleatoire(rng_test) for _ in range(150)]

        politiques = {
            "apprise": lambda s: self._episode(s, 0.0, rng_test, False),
            "gloutonne": _politique_gloutonne,
            "aleatoire": lambda s: _politique_aleatoire(s, rng_test),
        }
        mesures = {
            nom: {"recompense": [], "maitrise": [], "maitrise_ponderee": [],
                  "retention_ponderee": [], "stabilite": []}
            for nom in politiques
        }

        for nom, executer in politiques.items():
            for scenario in scenarios:
                total, m, stabilite = executer(_copier(scenario))
                poids = scenario["consensus"]

                # Retention : etat des acquis apres une periode sans travail.
                # Mesurer la maitrise a l'instant ou le parcours s'acheve
                # reviendrait a ignorer l'oubli que le modele simule — or c'est
                # bien ce qui reste quelques semaines plus tard qui compte.
                m_retenue = retention(m, stabilite)

                mesures[nom]["recompense"].append(total)
                mesures[nom]["maitrise"].append(float(np.mean(m)))
                # Maitrise ponderee par l'importance internationale : une
                # notion exigee par cinq referentiels ne vaut pas une notion
                # propre a un seul pays.
                mesures[nom]["maitrise_ponderee"].append(
                    float(np.sum(m * poids) / max(np.sum(poids), 1e-9))
                )
                mesures[nom]["retention_ponderee"].append(
                    float(np.sum(m_retenue * poids) / max(np.sum(poids), 1e-9))
                )
                mesures[nom]["stabilite"].append(float(np.mean(stabilite)))

        moyennes = {
            nom: {cle: round(float(np.mean(valeurs)), 3) for cle, valeurs in donnees.items()}
            for nom, donnees in mesures.items()
        }
        reference = moyennes["gloutonne"]

        self.infos = {
            "algorithme": "Apprentissage par renforcement model-based : TD(0) sur la valeur d'état + anticipation par le modèle d'acquisition",
            "formulation": (
                "a* = argmax_a [ r(s,a) + γ·V(résumé(s')) ]. La récompense immédiate et "
                "l'état suivant sont calculés exactement à partir du modèle "
                "d'acquisition ; seule V est apprise, par différence temporelle. "
                "La politique gloutonne est le cas particulier V ≡ 0."
            ),
            "nb_episodes": nb_episodes,
            "taux_apprentissage": TAUX_APPRENTISSAGE,
            "facteur_actualisation": FACTEUR_ACTUALISATION,
            "taille_table_v": int(np.prod(FORME_V)),
            "entrees_visitees": int(np.count_nonzero(self.V)),
            "duree_entrainement_s": round(time.time() - debut, 2),
            "recompense_moyenne_finale": round(float(np.mean(recompenses[-200:])), 3),
            "evaluation_comparative": {
                "protocole": (
                    "150 scénarios de test non vus pendant l'entraînement ; "
                    "politique apprise comparée à la politique gloutonne "
                    "(comportement de la version initiale de l'Agent 8) et à une "
                    "politique aléatoire."
                ),
                "metriques": {
                    "recompense": "Récompense cumulée — objectif effectivement optimisé.",
                    "maitrise_ponderee": (
                        "Maîtrise finale pondérée par l'importance internationale des "
                        "notions — grandeur visée par le système."
                    ),
                    "retention_ponderee": (
                        "Maîtrise pondérée conservée après 8 périodes sans travail — "
                        "ce qui reste réellement acquis, mesure pédagogiquement décisive."
                    ),
                    "maitrise": "Maîtrise finale moyenne non pondérée.",
                    "stabilite": "Consolidation moyenne des acquis en fin de parcours.",
                },
                "resultats": moyennes,
                "gains_vs_gloutonne_pct": {
                    cle: round(
                        100 * (moyennes["apprise"][cle] - reference[cle])
                        / max(1e-6, abs(reference[cle])), 1
                    )
                    for cle in ("recompense", "retention_ponderee", "maitrise_ponderee", "maitrise", "stabilite")
                },
            },
        }
        return self.infos

    # -- Deroulement d'un episode ------------------------------------------
    def _episode(self, scenario: dict, epsilon: float, rng, apprendre: bool):
        m = scenario["maitrise"].copy()
        consensus = scenario["consensus"]
        stabilite = np.zeros_like(m)
        budget_total = scenario["budget"]
        budget = budget_total
        total = 0.0

        while budget > 0:
            etat = _etat_courant(m, stabilite, scenario)
            ratio_budget = budget / budget_total
            resume_avant = resumer(m, stabilite, ratio_budget)
            q, recompenses, nouvelles = self.scores(etat, budget, budget_total)

            if apprendre and rng.random() < epsilon:
                i, t = _action_aleatoire_valide(recompenses, rng)
            else:
                i, t = divmod(int(np.argmax(q)), NB_INTERVENTIONS)

            r = float(recompenses[i, t])
            total += r
            _appliquer(m, stabilite, i, t, nouvelles)
            budget -= INTERVENTIONS[t]["cout_seances"]

            if apprendre:
                # TD(0) sur la valeur d'etat : V(s) <- V(s) + α[r + γV(s') − V(s)]
                if budget > 0:
                    resume_apres = resumer(m, stabilite, budget / budget_total)
                    cible = r + FACTEUR_ACTUALISATION * float(self.V[resume_apres])
                else:
                    cible = r + recompense_terminale(m, stabilite, consensus)
                self.V[resume_avant] += TAUX_APPRENTISSAGE * (cible - self.V[resume_avant])

        total += recompense_terminale(m, stabilite, consensus)
        return total, m, stabilite


# ---------------------------------------------------------------------------
# Scenarios synthetiques d'entrainement
# ---------------------------------------------------------------------------

def _scenario_aleatoire(rng) -> dict:
    """
    Genere un support de cours fictif : profil de maitrise initial, importance
    internationale des notions, gravite des ecarts et chaine de prerequis.

    La diversite de ces scenarios est ce qui permet a la politique de
    generaliser a un cours reel qu'elle n'a jamais vu.
    """
    n = int(rng.integers(10, 26))
    maitrise = np.clip(rng.beta(1.6, 3.0, size=n), 0.0, 1.0)
    consensus = np.clip(rng.beta(2.0, 2.0, size=n), 0.0, 1.0)
    # La gravite est correlee negativement a la maitrise, comme dans les
    # donnees reelles produites par l'Agent 6.
    gravite = np.clip(1.0 - maitrise + rng.normal(0, 0.12, size=n), 0.05, 1.0)
    # Chaine de prerequis : chaque notion depend de la precedente, sauf les
    # tetes de chapitre. Les chaines sont volontairement longues — c'est ce
    # qui rend l'anticipation necessaire : debloquer une notion amont peut
    # valoir bien plus que le gain immediat qu'elle rapporte.
    precedent = np.arange(-1, n - 1)
    tetes = rng.random(n) < 0.15
    precedent[tetes] = -1
    return {
        "maitrise": maitrise,
        "consensus": consensus,
        "gravite": gravite,
        "precedent": precedent,
        "budget": float(rng.integers(8, 21)),
    }


def _copier(scenario: dict) -> dict:
    copie = dict(scenario)
    copie["maitrise"] = scenario["maitrise"].copy()
    return copie


def _action_aleatoire_valide(recompenses: np.ndarray, rng) -> tuple[int, int]:
    """Tire une action au hasard parmi les actions non masquees."""
    valides = np.argwhere(recompenses > ACTION_INTERDITE / 2)
    if len(valides) == 0:
        return 0, CLES_INTERVENTIONS.index("theorie")
    i, t = valides[int(rng.integers(len(valides)))]
    return int(i), int(t)


def _maitrise_amont(maitrise: np.ndarray, precedent: np.ndarray) -> np.ndarray:
    """Maitrise du prerequis de chaque notion (1,0 s'il n'y en a pas)."""
    amont = np.ones_like(maitrise)
    valides = precedent >= 0
    amont[valides] = maitrise[precedent[valides]]
    return amont


def _etat_courant(maitrise: np.ndarray, stabilite: np.ndarray, scenario: dict) -> dict:
    """Assemble les caracteristiques d'etat exploitees par la politique."""
    return {
        "maitrise": maitrise,
        "prerequis": _maitrise_amont(maitrise, scenario["precedent"]),
        "consensus": scenario["consensus"],
        "gravite": scenario["gravite"],
        "potentiel": potentiel_aval(maitrise, scenario["consensus"], scenario["precedent"]),
        "stabilite": stabilite,
    }


def _appliquer(maitrise: np.ndarray, stabilite: np.ndarray, i: int, t: int,
               nouvelles: np.ndarray) -> None:
    """Applique une action sur place : maitrise acquise et trace consolidee."""
    maitrise[i] = nouvelles[i, t]
    stabilite[i] = min(STABILITE_MAX, stabilite[i] + INTERVENTIONS[t]["gain_stabilite"])


def retention(maitrise, stabilite):
    """Maitrise conservee apres `HORIZON_RETENTION` periodes sans pratique."""
    m = np.asarray(maitrise, dtype="float64").copy()
    for _ in range(HORIZON_RETENTION):
        m = oublier(m, stabilite)
    return m


def recompense_terminale(maitrise, stabilite, consensus) -> float:
    """
    Recompense de fin d'episode : ce que l'eleve retiendra reellement, pondere
    par l'importance internationale des notions.

    Sans ce terme, rien n'inciterait a consolider : enchainer des apports
    nouveaux maximiserait la maitrise instantanee, au prix de tout oublier.
    """
    poids = np.asarray(consensus, dtype="float64")
    conservee = retention(maitrise, stabilite)
    return PRIME_RETENTION * float(np.sum(conservee * poids) / max(np.sum(poids), 1e-9))


def _politique_gloutonne(scenario: dict):
    """
    Reference : choisir a chaque pas l'action de recompense immediate
    maximale, sans jamais anticiper. C'est exactement le comportement de la
    version initiale de l'Agent 8, et le cas particulier W ≡ 0 de la
    politique apprise.
    """
    m = scenario["maitrise"]
    stabilite = np.zeros_like(m)
    budget = scenario["budget"]
    total = 0.0
    while budget > 0:
        etat = _etat_courant(m, stabilite, scenario)
        recompenses, nouvelles = evaluer_toutes_actions(
            etat["maitrise"], etat["prerequis"], etat["consensus"], etat["gravite"]
        )
        i, t = divmod(int(np.argmax(recompenses)), NB_INTERVENTIONS)
        total += float(recompenses[i, t])
        _appliquer(m, stabilite, i, t, nouvelles)
        budget -= INTERVENTIONS[t]["cout_seances"]
    total += recompense_terminale(m, stabilite, scenario["consensus"])
    return total, m, stabilite


def _politique_aleatoire(scenario: dict, rng):
    """Reference basse : action tiree au hasard."""
    m = scenario["maitrise"]
    stabilite = np.zeros_like(m)
    budget = scenario["budget"]
    total = 0.0
    while budget > 0:
        etat = _etat_courant(m, stabilite, scenario)
        recompenses, nouvelles = evaluer_toutes_actions(
            etat["maitrise"], etat["prerequis"], etat["consensus"], etat["gravite"]
        )
        i, t = _action_aleatoire_valide(recompenses, rng)
        total += float(recompenses[i, t])
        _appliquer(m, stabilite, i, t, nouvelles)
        budget -= INTERVENTIONS[t]["cout_seances"]
    total += recompense_terminale(m, stabilite, scenario["consensus"])
    return total, m, stabilite


# ---------------------------------------------------------------------------
# Acces a la politique
# ---------------------------------------------------------------------------

def obtenir_politique() -> PolitiqueQLearning:
    """Entraine la politique une seule fois par processus."""
    global _POLITIQUE
    if _POLITIQUE is not None:
        return _POLITIQUE
    with _LOCK:
        if _POLITIQUE is None:
            politique = PolitiqueQLearning()
            politique.entrainer()
            _POLITIQUE = politique
    return _POLITIQUE


def planifier(notions_etat: list[dict], graphe: dict, budget_seances: float,
              max_etapes: int = 12, max_par_notion: int = 3) -> dict:
    """
    Construit le parcours d'amelioration d'un cours reel.

    `notions_etat` : une entree par notion de referentiel, contenant au moins
    `cle`, `maitrise`, `consensus`, `gravite`, plus les metadonnees
    d'affichage (`notion`, `pays`, `code`, `descriptif`).

    Retourne la sequence ordonnee d'interventions, la trajectoire de maitrise
    prevue seance apres seance, et la comparaison des politiques.
    """
    if not notions_etat:
        return {"disponible": False, "motif": "aucune notion de référentiel à planifier",
                "etapes": [], "courbe_progression": []}

    cles = [n["cle"] for n in notions_etat]
    position = {cle: i for i, cle in enumerate(cles)}
    m = np.array([float(n["maitrise"]) for n in notions_etat], dtype="float64")
    consensus = np.array([float(n["consensus"]) for n in notions_etat], dtype="float64")
    gravite = np.array([float(n["gravite"]) for n in notions_etat], dtype="float64")

    # Prerequis principal de chaque notion : on retient le plus proche, pour
    # rester coherent avec la structure en chaine vue a l'entrainement.
    precedent = np.full(len(cles), -1, dtype=int)
    for cle, amonts in (graphe.get("prerequis") or {}).items():
        if cle in position and amonts:
            for amont in amonts:
                if amont in position:
                    precedent[position[cle]] = position[amont]
                    break

    scenario = {
        "maitrise": m, "consensus": consensus, "gravite": gravite,
        "precedent": precedent, "budget": float(budget_seances),
    }
    politique = obtenir_politique()

    stabilite = np.zeros_like(m)
    budget = float(budget_seances)
    poids_total = max(float(consensus.sum()), 1e-9)

    def ponderee(vecteur):
        return float(np.sum(vecteur * consensus) / poids_total)

    maitrise_initiale = float(m.mean())
    ponderee_initiale = ponderee(m)
    courbe = [{
        "seance": 0.0,
        "maitrise_globale": round(maitrise_initiale, 3),
        "maitrise_ponderee": round(ponderee_initiale, 3),
    }]

    etapes = []
    seances_ecoulees = 0.0
    # Nombre d'interventions deja programmees par notion. Revenir plusieurs
    # fois sur une notion est souhaitable (effet d'espacement), mais une
    # progression annuelle ne peut pas y consacrer un quart de ses seances :
    # au-dela du plafond, la notion sort des candidats.
    compteur_notion = np.zeros(len(cles), dtype=int)

    while budget > 0 and len(etapes) < max_etapes:
        etat = _etat_courant(m, stabilite, scenario)
        q, recompenses, nouvelles = politique.scores(etat, budget, float(budget_seances))
        q = np.where(
            (compteur_notion >= max_par_notion)[:, None], ACTION_INTERDITE, q
        )
        if not np.any(q > ACTION_INTERDITE / 2):
            break
        i, t = divmod(int(np.argmax(q)), NB_INTERVENTIONS)
        compteur_notion[i] += 1

        intervention = INTERVENTIONS[t]
        avant = float(m[i])
        apres = float(nouvelles[i, t])
        prerequis_acquis = float(etat["prerequis"][i])
        source = notions_etat[i]

        _appliquer(m, stabilite, i, t, nouvelles)
        budget -= intervention["cout_seances"]
        seances_ecoulees += intervention["cout_seances"]

        cle_prerequis = (
            cles[precedent[i]] if precedent[i] >= 0 else None
        )
        etapes.append({
            "rang": len(etapes) + 1,
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
            "seance_cumulee": round(seances_ecoulees, 1),
            "maitrise_avant": round(avant, 3),
            "maitrise_apres_predite": round(apres, 3),
            "gain_predit": round(apres - avant, 3),
            "consensus": round(float(consensus[i]), 3),
            "gravite": round(float(gravite[i]), 3),
            "type_ecart": source.get("type_ecart", ""),
            "libelle_ecart": source.get("libelle_ecart", ""),
            "prerequis_acquis": round(prerequis_acquis, 3),
            "prerequis_bloquant": bool(prerequis_acquis < 0.50),
            "notion_prerequis": cle_prerequis,
            "recompense": round(float(recompenses[i, t]), 4),
            "justification": _justifier(source, intervention, avant, apres,
                                        prerequis_acquis, float(consensus[i])),
        })
        courbe.append({
            "seance": round(seances_ecoulees, 1),
            "maitrise_globale": round(float(m.mean()), 3),
            "maitrise_ponderee": round(ponderee(m), 3),
        })

    # Retention prevue une fois la sequence achevee.
    m_retenue = retention(m, stabilite)

    repartition: dict[str, int] = {}
    for etape in etapes:
        repartition[etape["intervention_nom"]] = repartition.get(etape["intervention_nom"], 0) + 1

    return {
        "disponible": True,
        "moteur": politique.infos.get("algorithme", "politique apprise"),
        "budget_seances": float(budget_seances),
        "seances_planifiees": round(seances_ecoulees, 1),
        "nb_etapes": len(etapes),
        "nb_notions_distinctes": len({e["cle"] for e in etapes}),
        "etat_initial": {
            "maitrise_globale": round(maitrise_initiale, 3),
            "maitrise_ponderee": round(ponderee_initiale, 3),
        },
        "etat_final_predit": {
            "maitrise_globale": round(float(m.mean()), 3),
            "maitrise_ponderee": round(ponderee(m), 3),
            "stabilite_moyenne": round(float(stabilite.mean()), 3),
            "retention_ponderee": round(ponderee(m_retenue), 3),
        },
        "gain_maitrise_ponderee": round(ponderee(m) - ponderee_initiale, 3),
        "etapes": etapes,
        "courbe_progression": courbe,
        "repartition_interventions": repartition,
        "graphe_prerequis": {
            "methode": graphe.get("methode"),
            "nb_aretes": graphe.get("nb_aretes", 0),
        },
        "modele_apprentissage": politique.infos,
    }


def _justifier(source: dict, intervention: dict, avant: float, apres: float,
               prerequis: float, consensus: float) -> str:
    """Explication en clair du choix effectue par le planificateur."""
    morceaux = []
    if consensus >= 0.60:
        morceaux.append("notion attendue par la majorité des référentiels comparés")
    elif consensus >= 0.35:
        morceaux.append("notion attendue par plusieurs référentiels")
    else:
        morceaux.append(f"notion propre au référentiel {source.get('pays', '')}")

    libelle = source.get("libelle_ecart") or source.get("type_ecart") or ""
    if libelle:
        morceaux.append(f"diagnostic : {str(libelle).lower()}")

    if prerequis < 0.50:
        morceaux.append(
            f"prérequis encore fragiles ({prerequis:.2f}) — l'apprentissage sera amorti"
        )
    else:
        morceaux.append(f"prérequis suffisamment acquis ({prerequis:.2f})")

    morceaux.append(
        f"{intervention['nom'].lower()} choisie pour ce niveau de maîtrise "
        f"({avant:.2f} → {apres:.2f} attendu)"
    )
    return " ; ".join(morceaux).capitalize() + "."


def politique_profonde():
    """
    Reseau Q profond entraine hors ligne, s'il a ete depose.

    L'artefact attendu est un dictionnaire contenant les poids et la
    description de l'architecture ; il consomme les memes caracteristiques
    que la table Q, sous forme continue.
    """
    artefact = model_registry.charger("dqn_planificateur")
    if artefact is None:
        return None
    try:
        import torch

        etat = artefact["state_dict"]
        dimensions = artefact.get("dimensions", [5, 64, 64, NB_INTERVENTIONS])
        couches = []
        for i in range(len(dimensions) - 1):
            couches.append(torch.nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                couches.append(torch.nn.ReLU())
        reseau = torch.nn.Sequential(*couches)
        reseau.load_state_dict(etat)
        reseau.eval()
        return reseau
    except Exception:
        return None
