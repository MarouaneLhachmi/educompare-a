"""
Agent 8 — Recommandations
==========================

Role (rapport de conception, section 3.3.2) : formuler, a partir des ecarts
identifies par l'Agent 6 et des evaluations produites par l'Agent 7, des
recommandations d'amelioration concretes, priorisees et rattachees a une
notion et a un referentiel precis.

Ce que cet agent produit desormais
-----------------------------------

La version initiale rendait une **liste d'ecarts classes**. Elle repondait a
« quelles notions manquent ? », jamais a « dans quel ordre les traiter, avec
quelle activite, et pour quel resultat attendu ? ». L'agent produit
maintenant un **parcours d'amelioration** : une sequence datee
d'interventions pedagogiques, chacune assortie de la theorie a ajouter, des
exercices gradues correspondants, du critere de reussite, et de la
trajectoire de maitrise prevue.

Trois sources de priorisation, strictement separees
----------------------------------------------------

L'exigence de separation des sources structure toute la sortie de cet agent.
**Trois sources independantes** proposent chacune leur ordre de priorite, et
chacune reste visible separement :

1. `parcours_algorithmique` — **planificateur par renforcement**, nos modeles
   seuls, aucun modele de langage. Le MDP + TD(0) de `services/rl_parcours`
   decide *quoi* traiter, *dans quel ordre*, *avec quel type d'activite*. La
   priorisation decoule du profil de maitrise mesure par l'Agent 7, du graphe
   de prerequis et de l'importance internationale de chaque notion.

2. `parcours_graphe` — **ordonnancement par centralite**, nos modeles seuls
   egalement, mais d'une nature radicalement differente. PageRank personnalise
   (`services/graphe_parcours`) sur le graphe de prerequis inverse, pondere
   par la gravite des ecarts.

3. `recommandations_gemini` — **le modele de langage seul**, libre de ses
   propositions, sans aucune contrainte issue de nos modeles.

Un quatrieme bloc, `contenu_pedagogique`, n'est pas une source : le modele de
langage y **habille** le parcours du planificateur RL. Il ne choisit rien, il
recoit des etapes deja decidees et redige la theorie et les exercices.

Pourquoi deux algorithmes locaux, et non deux variantes du meme
---------------------------------------------------------------

    Planificateur RL (bloc 1)           Centralite de graphe (bloc 2)
    ------------------------------      ------------------------------
    Decision sequentielle sous          Analyse structurelle, statique
    incertitude : MDP + TD(0)           PageRank, non supervise
    « Quelle action maximise le         « Quelles notions debloquent le
    gain cumule attendu ? »             plus d'autres notions ? »
    Simule l'acquisition seance         Aucun modele d'eleve, aucune
    apres seance (modele type BKT)      simulation temporelle
    Entree decisive : maitrise          Entree decisive : topologie des
    estimee + budget de seances         prerequis + gravite de l'ecart
    Entraine sur un simulateur          Ne s'entraine sur rien : le score
                                        se calcule, il ne s'apprend pas

Les deux peuvent donc **diverger reellement**, et pas seulement par le bruit :
le planificateur privilegie une notion grave dont la remediation rapporte
immediatement, l'ordonnancement par graphe privilegie une notion peut-etre
moins grave mais dont depend une grappe entiere du programme. Ce desaccord est
l'information qu'une source unique ne peut pas produire.

Le choix de PageRank tient aussi a la contrainte de donnees du projet : tres
peu d'analyses, aucune etiquette de priorite produite par un humain. Un modele
multi-criteres supervise aurait suppose des etiquettes inventees — exactement
ce que le projet s'interdit. PageRank ne s'entraine sur rien, et le projet
emploie deja cette famille d'algorithmes avec TextRank dans le module Rapport
et Restitution.

La confrontation a trois voix
------------------------------

`confrontation` compare desormais les trois sources. Une notion citee par les
deux algorithmes locaux ET par le modele de langage constitue un diagnostic
**particulierement solide** : trois methodes sans rien de commun dans leur
raisonnement aboutissent au meme point. A l'inverse, une notion retenue par
une seule source est signalee comme telle, a l'arbitrage de l'enseignant.

Entree : sorties des Agents 2, 3, 4, 6 et 7 + notions de reference
Sortie : deux parcours algorithmiques, contenu pedagogique, recommandations
         LLM, confrontation a trois voix
"""

import re
import unicodedata

import numpy as np

from app.config import Config
from app.services import (
    gemini_client, graphe_parcours, prerequis, reranking, rl_parcours,
)

# Seuil de similarite au-dela duquel deux notions de pays differents sont
# considerees comme portant sur le meme apprentissage.
SEUIL_EQUIVALENCE = 0.72
# Budget de seances par defaut du parcours propose.
SEANCES_PAR_ETAPE = 1.2


# ---------------------------------------------------------------------------
# Preparation de l'etat initial du planificateur
# ---------------------------------------------------------------------------

def _consensus_international(notions_reference: list[dict], vecteurs) -> dict:
    """
    Importance internationale de chaque notion : part des referentiels
    comparés qui exigent un apprentissage equivalent.

    Deux notions redigees dans des langues differentes (« Fractions simples
    et nombres decimaux » / « Fractions, decimals and percentages ») doivent
    etre reconnues comme equivalentes : c'est precisement ce que permet
    l'espace vectoriel multilingue de l'Agent 4.
    """
    codes = [n["code"] for n in notions_reference]
    pays_total = len(set(codes)) or 1

    if vecteurs is None or len(vecteurs) != len(notions_reference):
        # Sans vecteurs, on ne peut pas apparier les langues : chaque notion
        # ne compte que pour son propre referentiel.
        return {
            f"{n['code']}::{n['notion']}": 1.0 / pays_total for n in notions_reference
        }

    similarites = np.asarray(vecteurs) @ np.asarray(vecteurs).T
    consensus = {}
    for i, notion in enumerate(notions_reference):
        equivalents = np.where(similarites[i] >= SEUIL_EQUIVALENCE)[0]
        pays_couverts = {codes[j] for j in equivalents} | {codes[i]}
        consensus[f"{notion['code']}::{notion['notion']}"] = len(pays_couverts) / pays_total
    return consensus


def _etat_notions(agent7: dict, consensus: dict) -> list[dict]:
    """Construit l'etat initial du MDP a partir du profil de maitrise."""
    etat = []
    for entree in (agent7.get("profil_maitrise") or {}).get("par_notion", []):
        cle = entree["cle"]
        etat.append({
            "cle": cle,
            "notion": entree["notion"],
            "descriptif": entree.get("descriptif", ""),
            "pays": entree["pays"],
            "code": entree["code"],
            "maitrise": float(entree["maitrise"]),
            "consensus": float(consensus.get(cle, 0.2)),
            "gravite": rl_parcours.GRAVITE_ECART.get(entree.get("type_ecart"), 0.5),
            "type_ecart": entree.get("type_ecart", ""),
            "libelle_ecart": entree.get("libelle_ecart", ""),
            "chapitre_correspondant": entree.get("chapitre_correspondant"),
        })
    return etat


# ---------------------------------------------------------------------------
# Bloc 2 — contenu pedagogique redige par le modele de langage
# ---------------------------------------------------------------------------

PROMPT_CONTENU = """Tu es un ingenieur pedagogique qui redige du materiel de cours.

Cours : {titre_cours} — {matiere}, niveau « {niveau} ».

Un planificateur automatique a DEJA decide quoi traiter et dans quel ordre.
Tu ne dois RIEN reordonner ni proposer d'autres notions : ta seule mission est
de rediger le contenu de chacune des etapes ci-dessous, telles qu'elles sont.

Etapes a rediger :
{etapes}

Pour chaque etape, produis :
- un apport theorique adapte au niveau de maitrise indique ;
- trois exercices de difficulte croissante (application directe, application
  guidee, transfert), avec leur reponse attendue ;
- un critere de reussite chiffre permettant de valider l'etape ;
- l'erreur que les eleves commettent le plus souvent sur cette notion.

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour :
{{
  "etapes": [
    {{
      "rang": 1,
      "notion": "reprends exactement l'intitule fourni",
      "titre_seance": "titre de la seance (max 12 mots)",
      "theorie": {{
        "rappel": "ce qu'il faut avoir compris avant (2 phrases max)",
        "apport": "l'apport theorique proprement dit (4 a 6 phrases)",
        "exemple": "un exemple filé, entierement resolu"
      }},
      "exercices": [
        {{"niveau": "application directe", "enonce": "...", "reponse": "..."}},
        {{"niveau": "application guidee", "enonce": "...", "reponse": "..."}},
        {{"niveau": "transfert", "enonce": "...", "reponse": "..."}}
      ],
      "critere_reussite": "critere chiffre, par exemple 4 exercices sur 5",
      "erreur_frequente": "l'erreur typique et comment la prevenir"
    }}
  ]
}}

Contraintes : francais, contenu reellement adapte au niveau « {niveau} »,
enonces courts et concrets, calculs justes.
"""


def _rediger_contenu(parcours: dict, agent2: dict, matiere: str, niveau: str,
                     nb_etapes: int) -> dict:
    """Fait rediger par le modele de langage le contenu des premieres etapes."""
    etapes = parcours.get("etapes", [])[:nb_etapes]
    if not etapes:
        return {"disponible": False, "erreur": None, "etapes": [],
                "nb_etapes_redigees": 0}

    description = "\n".join(
        f"{e['rang']}. Notion « {e['notion']} » ({e['pays']}) — "
        f"activite demandee : {e['intervention_nom']} — "
        f"niveau de Bloom vise : {e['bloom_cible']} — "
        f"maitrise actuelle estimee : {e['maitrise_avant']:.2f}/1 — "
        f"diagnostic : {e.get('libelle_ecart') or e.get('type_ecart')}"
        + (f" — descriptif du referentiel : {e['descriptif']}" if e.get("descriptif") else "")
        for e in etapes
    )

    prompt = PROMPT_CONTENU.format(
        titre_cours=agent2.get("titre_cours", ""),
        matiere=matiere,
        niveau=niveau,
        etapes=description,
    )

    try:
        donnees = gemini_client.generate_json(prompt, agent="agent8_contenu", temperature=0.5)
        redigees = donnees.get("etapes") if isinstance(donnees, dict) else None
        if not redigees:
            raise ValueError("structure JSON inattendue")
    except Exception as exc:
        return {
            "disponible": False,
            "erreur": str(exc),
            "etapes": [_gabarit_contenu(e) for e in etapes],
            "nb_etapes_redigees": len(etapes),
            "source": "gabarit_deterministe",
        }

    # Reappariement par rang : le modele peut renvoyer les etapes desordonnees.
    par_rang = {}
    for element in redigees:
        if isinstance(element, dict):
            try:
                par_rang[int(element.get("rang", 0))] = element
            except (TypeError, ValueError):
                continue

    resultat = []
    for index, etape in enumerate(etapes):
        contenu = par_rang.get(etape["rang"])
        if contenu is None and index < len(redigees):
            contenu = redigees[index] if isinstance(redigees[index], dict) else None
        resultat.append(_normaliser_contenu(etape, contenu))

    return {
        "disponible": True,
        "erreur": None,
        "etapes": resultat,
        "nb_etapes_redigees": len(resultat),
        "source": "gemini",
    }


def _normaliser_contenu(etape: dict, contenu: dict | None) -> dict:
    contenu = contenu or {}
    theorie = contenu.get("theorie") if isinstance(contenu.get("theorie"), dict) else {}
    exercices = [
        {
            "niveau": str(x.get("niveau", "")).strip(),
            "enonce": str(x.get("enonce", "")).strip(),
            "reponse": str(x.get("reponse", "")).strip(),
        }
        for x in (contenu.get("exercices") or [])
        if isinstance(x, dict) and str(x.get("enonce", "")).strip()
    ]
    return {
        "rang": etape["rang"],
        "cle": etape["cle"],
        "notion": etape["notion"],
        "intervention_nom": etape["intervention_nom"],
        "bloom_cible": etape["bloom_cible"],
        "titre_seance": str(contenu.get("titre_seance") or etape["intervention_nom"]).strip(),
        "theorie": {
            "rappel": str(theorie.get("rappel", "")).strip(),
            "apport": str(theorie.get("apport", "")).strip(),
            "exemple": str(theorie.get("exemple", "")).strip(),
        },
        "exercices": exercices,
        "critere_reussite": str(contenu.get("critere_reussite", "")).strip(),
        "erreur_frequente": str(contenu.get("erreur_frequente", "")).strip(),
    }


def _gabarit_contenu(etape: dict) -> dict:
    """Contenu minimal produit sans modele de langage."""
    return {
        "rang": etape["rang"],
        "cle": etape["cle"],
        "notion": etape["notion"],
        "intervention_nom": etape["intervention_nom"],
        "bloom_cible": etape["bloom_cible"],
        "titre_seance": f"{etape['intervention_nom']} — {etape['notion']}",
        "theorie": {
            "rappel": "",
            "apport": etape.get("descriptif")
            or f"Traiter la notion « {etape['notion']} » attendue par le référentiel "
               f"{etape['pays']}.",
            "exemple": "",
        },
        "exercices": [],
        "critere_reussite": "",
        "erreur_frequente": "",
    }


def _verifier_coherence(contenu: dict, parcours: dict) -> dict:
    """
    Verifie que les exercices generes portent bien sur la notion visee.

    Le meme cross-encodeur que celui de l'Agent 6 est reutilise ici, en sens
    inverse : il avait servi a detecter ce que le cours ne traitait pas, il
    sert maintenant a valider que ce qui est propose traite bien la notion.

    Le test est **contrastif**, et non fonde sur un seuil absolu : chaque
    enonce est confronte a sa notion cible ET aux autres notions du parcours.
    L'exercice est juge coherent si sa notion cible arrive en tete. Ce choix
    est important — la calibration absolue etablie pour comparer des *unites
    de cours* a des notions ne vaut pas pour de courts enonces d'exercices,
    dont les logits se situent sur une toute autre plage. Un test par rang est
    insensible a ce decalage d'echelle.
    """
    exercices, notions_cibles = [], []
    for etape in contenu.get("etapes", []):
        for exercice in etape.get("exercices", []):
            if exercice["enonce"]:
                exercices.append((etape["rang"], etape["notion"], exercice["niveau"],
                                  exercice["enonce"]))
        if etape["notion"] not in notions_cibles:
            notions_cibles.append(etape["notion"])

    if not exercices:
        return {"applique": False, "motif": "aucun exercice généré"}
    if len(notions_cibles) < 2:
        return {"applique": False,
                "motif": "un seul thème dans le parcours — test contrastif impossible"}

    # Chaque enonce est confronte a toutes les notions du parcours.
    paires = [
        (enonce[:600], notion[:300])
        for (_, _, _, enonce) in exercices
        for notion in notions_cibles
    ]
    logits = reranking.scorer_paires(paires)
    if logits is None:
        return {"applique": False, "motif": "cross-encodeur indisponible"}

    matrice = np.asarray(logits).reshape(len(exercices), len(notions_cibles))
    index_cible = {notion: i for i, notion in enumerate(notions_cibles)}

    suspects, rangs = [], []
    for ligne, (rang, notion, niveau, enonce) in zip(matrice, exercices):
        cible = index_cible[notion]
        # Rang de la notion cible parmi toutes les notions (1 = en tete).
        position = int(np.sum(ligne > ligne[cible])) + 1
        rangs.append(position)
        if position > 1:
            meilleure = notions_cibles[int(np.argmax(ligne))]
            suspects.append({
                "rang_etape": rang,
                "niveau": niveau,
                "notion_visee": notion,
                "notion_plus_proche": meilleure,
                "position_cible": position,
                "extrait": enonce[:120],
            })

    part_correcte = float(np.mean([r == 1 for r in rangs]))
    return {
        "applique": True,
        "methode": "test contrastif : la notion visée doit arriver en tête des notions du parcours",
        "nb_exercices_verifies": len(exercices),
        "nb_notions_comparees": len(notions_cibles),
        "part_correctement_cibles_pct": round(100 * part_correcte, 1),
        "rang_moyen_cible": round(float(np.mean(rangs)), 2),
        "nb_suspects": len(suspects),
        "suspects": suspects[:6],
        "verdict": (
            "Chaque exercice généré est bien plus proche de sa notion cible que de "
            "toute autre notion du parcours."
            if not suspects
            else f"{len(suspects)} exercice(s) sur {len(exercices)} sont plus proches d'une "
                 f"autre notion du parcours — à relire."
        ),
    }


# ---------------------------------------------------------------------------
# Bloc 3 — recommandations libres du modele de langage
# ---------------------------------------------------------------------------

PROMPT_LIBRE = """Tu es un expert en ingenierie pedagogique.

Voici le diagnostic automatique d'un cours de {matiere} destine au niveau
« {niveau} », compare a des referentiels internationaux.

Titre : {titre_cours}
Resume : {resume}
Note d'alignement : {note}/100 ({maturite})
Chapitres existants : {chapitres}

Couverture mesuree par referentiel :
{couverture}

Notions non couvertes : {manquantes}
Notions traitees trop brievement : {superficielles}
Profondeur cognitive du support (taxonomie de Bloom) : {bloom}

Formule TES PROPRES recommandations d'amelioration, librement, en pedagogue.
N'essaie pas de deviner ce qu'un algorithme aurait propose : donne ton avis.

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour :
{{
  "recommandations": [
    {{
      "titre": "action a mener, a l'infinitif (max 12 mots)",
      "notion_visee": "la notion concernee, ou 'transversal'",
      "priorite": "critique" | "haute" | "moyenne",
      "argument": "2 phrases justifiant pedagogiquement cette recommandation",
      "mise_en_oeuvre": "comment s'y prendre concretement"
    }}
  ],
  "angle_mort": "un point que les indicateurs chiffres ne peuvent pas capter",
  "synthese": "2 phrases resumant ta strategie d'amelioration"
}}

Contraintes : francais, 5 a 7 recommandations, formulations concretes.
"""


def _recommandations_libres(agent2: dict, agent6: dict, agent7: dict,
                            matiere: str, niveau: str) -> dict:
    couverture = "\n".join(
        f"- {p['pays']} : {p['taux_couverture_pct']} % de couverture, "
        f"{p.get('taux_traitement_approfondi_pct', 0)} % traité en profondeur"
        for p in (agent6.get("par_pays") or {}).values()
    ) or "(aucun référentiel)"
    manquantes = ", ".join(
        n["notion"] for n in agent6.get("notions_manquantes", [])[:14]
    ) or "(aucune)"
    superficielles = ", ".join(
        n["notion"] for n in agent6.get("notions_superficielles", [])[:10]
    ) or "(aucune)"

    prompt = PROMPT_LIBRE.format(
        matiere=matiere,
        niveau=niveau,
        titre_cours=agent2.get("titre_cours", ""),
        resume=agent2.get("resume", ""),
        note=agent7.get("note_globale", 0),
        maturite=agent7.get("niveau_maturite", ""),
        chapitres=", ".join(c["titre"] for c in agent2.get("chapitres", [])[:12]) or "(aucun)",
        couverture=couverture,
        manquantes=manquantes,
        superficielles=superficielles,
        bloom=(agent7.get("bloom") or {}).get("niveau_max_atteint") or "non déterminée",
    )

    try:
        donnees = gemini_client.generate_json(prompt, agent="agent8_libre", temperature=0.6)
        recommandations = donnees.get("recommandations") if isinstance(donnees, dict) else None
        if not recommandations:
            raise ValueError("structure JSON inattendue")
        return {
            "disponible": True,
            "erreur": None,
            "recommandations": [
                {
                    "titre": str(r.get("titre", "")).strip(),
                    "notion_visee": str(r.get("notion_visee", "")).strip(),
                    "priorite": str(r.get("priorite", "moyenne")).strip().lower(),
                    "argument": str(r.get("argument", "")).strip(),
                    "mise_en_oeuvre": str(r.get("mise_en_oeuvre", "")).strip(),
                }
                for r in recommandations if isinstance(r, dict)
            ],
            "angle_mort": str(donnees.get("angle_mort", "")).strip(),
            "synthese": str(donnees.get("synthese", "")).strip(),
            "avertissement": (
                "Suggestions formulées librement par le modèle de langage. Elles ne sont "
                "issues d'aucun de nos modèles et n'ont pas été vérifiées par le "
                "planificateur : à lire comme un second avis, pas comme un résultat mesuré."
            ),
        }
    except Exception as exc:
        return {
            "disponible": False,
            "erreur": str(exc),
            "recommandations": [],
            "angle_mort": "",
            "synthese": "",
            "avertissement": "Le modèle de langage n'a pas pu être sollicité.",
        }


# ---------------------------------------------------------------------------
# Bloc 4 — confrontation des deux sources
# ---------------------------------------------------------------------------

def _normaliser(texte: str) -> str:
    sans_accents = "".join(
        c for c in unicodedata.normalize("NFKD", (texte or "").lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9 ]", " ", sans_accents)


def _mots_cles(texte: str) -> set[str]:
    return {m for m in _normaliser(texte).split() if len(m) >= 5}


def _apparier_gemini(intitules: set[str], libres: dict) -> dict:
    """
    Rattache chaque recommandation du modele de langage a une notion.

    L'appariement est lexical (recouvrement de mots significatifs) :
    volontairement simple et verifiable, l'objectif n'etant pas de trancher
    mais de signaler ou les sources se rejoignent.
    """
    apparie: dict[str, dict] = {}
    for reco in libres.get("recommandations", []):
        mots_reco = _mots_cles(reco.get("notion_visee", "") + " " + reco.get("titre", ""))
        meilleure, score = None, 0.0
        for intitule in intitules:
            mots_notion = _mots_cles(intitule)
            if not mots_notion:
                continue
            recouvrement = len(mots_reco & mots_notion) / len(mots_notion)
            if recouvrement > score:
                meilleure, score = intitule, recouvrement
        if meilleure and score >= 0.34:
            # Une notion peut etre visee par plusieurs recommandations : on
            # garde la mieux appariee.
            precedent = apparie.get(meilleure)
            if precedent is None or score > precedent["recouvrement"]:
                apparie[meilleure] = {"reco": reco, "recouvrement": round(score, 2)}
    return apparie


def _confronter(parcours: dict, parcours_graphe: dict, libres: dict) -> dict:
    """
    Confronte les TROIS sources de priorisation.

    Le principe est le meme qu'auparavant — signaler ou les sources se
    rejoignent et ou elles divergent — mais avec trois voix au lieu de deux,
    ce qui change la lecture : un accord entre le planificateur par
    renforcement, l'ordonnancement par graphe et le modele de langage est
    bien plus fort qu'un accord entre deux sources seulement. Ces trois
    methodes n'ont rien de commun dans leur raisonnement ; leur convergence
    n'a donc pas d'explication triviale.
    """
    etapes_rl = {e["notion"]: e for e in (parcours.get("etapes") or [])}
    etapes_graphe = {e["notion"]: e for e in (parcours_graphe.get("etapes") or [])}
    gemini_disponible = bool(libres.get("disponible"))

    sources_actives = sum([bool(etapes_rl), bool(etapes_graphe), gemini_disponible])
    if sources_actives < 2:
        return {
            "disponible": False,
            "nb_sources": sources_actives,
            "accords_complets": [], "accords_partiels": [], "isolees": [],
            "convergences": [], "specifiques_algorithme": [], "specifiques_gemini": [],
            "lecture": "Confrontation impossible : moins de deux sources disponibles.",
        }

    intitules = set(etapes_rl) | set(etapes_graphe)
    apparie = _apparier_gemini(intitules, libres) if gemini_disponible else {}

    accords_complets, accords_partiels, isolees = [], [], []
    for intitule in sorted(intitules):
        rl = etapes_rl.get(intitule)
        graphe = etapes_graphe.get(intitule)
        gemini = apparie.get(intitule)

        sources = []
        if rl:
            sources.append("planificateur")
        if graphe:
            sources.append("graphe")
        if gemini:
            sources.append("gemini")

        entree = {
            "notion": intitule,
            "sources": sources,
            "nb_sources": len(sources),
            "rang_planificateur": rl["rang"] if rl else None,
            "rang_graphe": graphe["rang"] if graphe else None,
            "intervention_planificateur": rl["intervention_nom"] if rl else None,
            "intervention_graphe": graphe["intervention_nom"] if graphe else None,
            "centralite": graphe.get("centralite") if graphe else None,
            "nb_notions_dependantes": graphe.get("nb_notions_dependantes") if graphe else None,
            "recommandation_gemini": gemini["reco"].get("titre") if gemini else None,
            "recouvrement": gemini["recouvrement"] if gemini else None,
            "ecart_de_rang": (
                abs(rl["rang"] - graphe["rang"]) if rl and graphe else None
            ),
        }

        # Un accord est « complet » quand TOUTES les sources disponibles
        # retiennent la notion — pas seulement deux d'entre trois.
        if len(sources) == sources_actives:
            accords_complets.append(entree)
        elif len(sources) >= 2:
            accords_partiels.append(entree)
        else:
            isolees.append(entree)

    # Les deux algorithmes locaux sont comparables rang a rang : le modele de
    # langage, lui, ne produit pas d'ordre exploitable.
    communes_locales = [
        e for e in accords_complets + accords_partiels
        if e["rang_planificateur"] is not None and e["rang_graphe"] is not None
    ]
    ecart_moyen = (
        round(sum(e["ecart_de_rang"] for e in communes_locales) / len(communes_locales), 1)
        if communes_locales else None
    )

    specifiques_gemini = [
        {"titre": r.get("titre"), "notion_visee": r.get("notion_visee"),
         "priorite": r.get("priorite")}
        for r in libres.get("recommandations", [])
        if not any(v["reco"] is r for v in apparie.values())
    ] if gemini_disponible else []

    total = len(intitules) or 1
    taux_complet = len(accords_complets) / total

    if taux_complet >= 0.35:
        lecture = (
            f"{len(accords_complets)} notion(s) sont retenues par les "
            f"{sources_actives} sources : sur celles-ci le diagnostic est "
            "particulièrement solide, trois méthodes sans rien de commun dans "
            "leur raisonnement aboutissant au même point."
        )
    elif accords_complets:
        lecture = (
            f"Seules {len(accords_complets)} notion(s) font l'unanimité des "
            f"{sources_actives} sources. Les autres relèvent de l'arbitrage : "
            "les sources ne mesurent pas la même chose."
        )
    else:
        lecture = (
            "Aucune notion ne fait l'unanimité des sources. Le désaccord est "
            "lui-même une information : il signale un cours dont les priorités "
            "dépendent fortement du critère retenu."
        )

    return {
        "disponible": True,
        "nb_sources": sources_actives,
        "sources": {
            "planificateur": bool(etapes_rl),
            "graphe": bool(etapes_graphe),
            "gemini": gemini_disponible,
        },
        "accords_complets": accords_complets,
        "accords_partiels": accords_partiels,
        "isolees": isolees,
        "nb_accords_complets": len(accords_complets),
        "taux_accord_complet_pct": round(100 * taux_complet, 1),
        "ecart_moyen_de_rang_entre_algorithmes": ecart_moyen,
        "specifiques_gemini": specifiques_gemini,
        "lecture": lecture,
        "methode": (
            "appariement lexical sur les mots significatifs des intitulés ; "
            "les deux parcours algorithmiques sont comparés par leur intitulé exact"
        ),
        # --- Compatibilite ascendante ------------------------------------
        # L'export PDF et les tests existants consomment ces trois cles ;
        # elles restent alimentees avec la lecture a deux voix
        # (algorithmes locaux contre modele de langage).
        "convergences": [
            {
                "notion": e["notion"],
                "recommandation_gemini": e["recommandation_gemini"],
                "intervention_algorithmique": (
                    e["intervention_planificateur"] or e["intervention_graphe"]
                ),
                "rang_parcours": e["rang_planificateur"] or e["rang_graphe"],
                "recouvrement": e["recouvrement"],
            }
            for e in accords_complets + accords_partiels
            if e["recommandation_gemini"]
        ],
        "specifiques_algorithme": [
            {
                "notion": e["notion"],
                "rang": e["rang_planificateur"] or e["rang_graphe"],
                "intervention": e["intervention_planificateur"] or e["intervention_graphe"],
                "consensus": (etapes_rl.get(e["notion"]) or
                              etapes_graphe.get(e["notion"], {})).get("consensus"),
            }
            for e in accords_partiels + isolees
            if not e["recommandation_gemini"]
        ],
        "taux_convergence_pct": round(
            100 * len([e for e in accords_complets + accords_partiels
                       if e["recommandation_gemini"]]) / total, 1
        ),
    }


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def process(agent2: dict, agent3: dict, agent4: dict, agent6: dict, agent7: dict,
            notions_reference: list[dict], matiere: str, niveau: str) -> dict:
    """Execute l'Agent 8 : parcours algorithmique, contenu rédigé, avis libre."""
    # --- Bloc 1 : parcours decide par nos modeles ------------------------
    consensus = _consensus_international(
        notions_reference, agent4.get("_vecteurs_referentiel")
    )
    graphe = prerequis.construire(notions_reference, agent4.get("_vecteurs_referentiel"))
    etat = _etat_notions(agent7, consensus)

    max_etapes = Config.PARCOURS_MAX_ETAPES
    parcours = rl_parcours.planifier(
        etat, graphe,
        budget_seances=max_etapes * SEANCES_PAR_ETAPE,
        max_etapes=max_etapes,
    )

    # --- Source 2 : ordonnancement par centralite de graphe ---------------
    # Deuxieme algorithme local, calcule sur le MEME etat et le MEME graphe
    # que le planificateur, mais selon une logique sans rapport : PageRank
    # mesure l'effet de levier structurel d'une notion, la ou le planificateur
    # optimise un gain d'apprentissage simule. Leur desaccord est informatif.
    # Il ne leve jamais : un echec ici ne doit pas priver l'analyse du reste.
    try:
        parcours_graphe = graphe_parcours.planifier(etat, graphe, max_etapes=max_etapes)
    except Exception as exc:  # pragma: no cover - filet de securite
        parcours_graphe = {
            "disponible": False,
            "motif": f"ordonnancement par graphe indisponible : {str(exc)[:120]}",
            "etapes": [], "nb_etapes": 0,
        }

    # --- Contenu pedagogique redige ---------------------------------------
    # Il continue de porter sur le parcours du planificateur, comme avant :
    # habiller deux parcours doublerait le cout sans rien clarifier.
    contenu = _rediger_contenu(
        parcours, agent2, matiere, niveau, Config.PARCOURS_ETAPES_REDIGEES
    )
    coherence = _verifier_coherence(contenu, parcours)

    # --- Source 3 : recommandations libres --------------------------------
    libres = _recommandations_libres(agent2, agent6, agent7, matiere, niveau)

    # --- Confrontation des trois sources ----------------------------------
    confrontation = _confronter(parcours, parcours_graphe, libres)

    # --- Vue compatible avec l'existant -----------------------------------
    # Le rapport et l'export PDF consomment historiquement une liste plate de
    # recommandations : on la derive du parcours pour ne rien casser.
    recommandations = [
        {
            "rang": e["rang"],
            "notion": e["notion"],
            "priorite": _priorite(e),
            "score_priorite": round(min(1.0, e["consensus"] * 0.6 + e["gravite"] * 0.4), 3),
            "referentiels": [e["pays"]],
            "nb_referentiels": 1,
            "statut_actuel": e.get("libelle_ecart") or e.get("type_ecart", ""),
            "score_similarite": e["maitrise_avant"],
            "effort_estime": f"{e['cout_seances']:g} séance(s)",
            "titre": f"{e['intervention_nom']} — {e['notion']}",
            "action_detaillee": e["justification"],
            "exemple_activite": e["intervention_description"],
            "beneficie_a": e["bloom_cible"],
        }
        for e in parcours.get("etapes", [])
    ]

    return {
        "parcours_algorithmique": parcours,
        "parcours_graphe": parcours_graphe,
        "moteurs_locaux": [
            {
                "cle": "parcours_algorithmique",
                "nom": "Planificateur par renforcement",
                "algorithme": parcours.get("moteur", "politique apprise"),
                "nature": "Apprentissage par renforcement (MDP + TD(0))",
                "disponible": bool(parcours.get("disponible")),
                "nb_etapes": parcours.get("nb_etapes", 0),
                "question": "Quelle action maximise le gain d'apprentissage cumulé ?",
            },
            {
                "cle": "parcours_graphe",
                "nom": "Ordonnancement par centralité",
                "algorithme": parcours_graphe.get("moteur", ""),
                "nature": "Analyse structurelle de graphe (non supervisée)",
                "disponible": bool(parcours_graphe.get("disponible")),
                "nb_etapes": parcours_graphe.get("nb_etapes", 0),
                "question": "Quelles notions débloquent le plus d'autres notions ?",
            },
        ],
        "contenu_pedagogique": contenu,
        "controle_coherence": coherence,
        "recommandations_gemini": libres,
        "confrontation": confrontation,
        "consensus_par_notion": consensus,
        "graphe_prerequis": graphe,
        # --- compatibilite ascendante ---
        "recommandations": recommandations,
        "nb_ecarts_analyses": len(parcours.get("etapes", [])),
        "source_redaction": "gemini" if contenu.get("disponible") else "repli_deterministe",
        "synthese": libres.get("synthese", "") or parcours.get("moteur", ""),
        "plan_action": _plan_action(parcours),
        "erreur": contenu.get("erreur") or libres.get("erreur"),
    }


def _priorite(etape: dict) -> str:
    score = etape["consensus"] * 0.6 + etape["gravite"] * 0.4
    if score >= 0.75:
        return "Critique"
    if score >= 0.55:
        return "Haute"
    if score >= 0.35:
        return "Moyenne"
    return "Faible"


def _plan_action(parcours: dict) -> dict:
    """Repartit les etapes du parcours en trois horizons temporels."""
    etapes = parcours.get("etapes", [])
    seuil_court = 4.0
    seuil_moyen = 9.0
    return {
        "court_terme": [
            f"Séance {e['seance_cumulee']:g} — {e['intervention_nom']} : {e['notion']}"
            for e in etapes if e["seance_cumulee"] <= seuil_court
        ][:4],
        "moyen_terme": [
            f"Séance {e['seance_cumulee']:g} — {e['intervention_nom']} : {e['notion']}"
            for e in etapes if seuil_court < e["seance_cumulee"] <= seuil_moyen
        ][:4],
        "long_terme": [
            f"Séance {e['seance_cumulee']:g} — {e['intervention_nom']} : {e['notion']}"
            for e in etapes if e["seance_cumulee"] > seuil_moyen
        ][:4],
    }
