"""
Agent 6 — Comparaison
======================

Role (rapport de conception, section 3.3.2) : confronter le contenu du cours
aux referentiels identifies par l'Agent 5, afin de produire un score de
similarite ainsi qu'une cartographie des notions communes, manquantes et
excedentaires.

Architecture en trois etages : **Rappel -> Precision -> Decision**
------------------------------------------------------------------

La version initiale de cet agent decidait de la couverture d'une notion a
partir du seul score cosinus du bi-encodeur, compare a des seuils fixes. Cette
approche souffre de deux defauts structurels :

1. Le bi-encodeur encode le cours et la notion **separement**. Il mesure donc
   une proximite *thematique*, pas un *enseignement effectif* : un cours qui
   evoque les fractions obtient un score eleve face a « conversion fraction /
   pourcentage » sans jamais enseigner cette conversion. D'ou une
   surestimation systematique de la couverture.
2. Des seuils fixes (0,50 / 0,33) ne sont ni calibres ni interpretables : un
   score de 0,49 et un score de 0,51 recoivent des verdicts opposes alors que
   rien ne distingue reellement les deux situations.

La chaine est donc desormais :

    Etage 1 — Rappel      Bi-encodeur + FAISS (Agents 4 et 5)
                          -> k unites de cours candidates par notion.
                          Rapide, pre-indexable, oriente rappel.

    Etage 2 — Precision   Cross-encodeur (Deep Learning)
                          -> re-score chaque paire (unite, notion) en une
                          seule passe d'attention conjointe. Non
                          pre-indexable donc lent, mais nettement plus
                          discriminant. C'est lui qui ecarte les faux
                          positifs de l'etage 1.

    Etage 3 — Decision    Classifieur calibre (Machine Learning)
                          -> fusionne huit caracteristiques en une
                          **probabilite de couverture** calibree, assortie
                          d'une zone d'incertitude explicite et de la
                          **preuve** textuelle qui justifie le verdict.

Repli en trois niveaux, conformement a la discipline du projet :
classifieur entraine -> fusion logistique reglee a la main -> seuils
d'origine. L'agent aboutit toujours.

L'analyse qualitative par modele de langage est conservee, mais son role
change : elle n'estime plus la couverture (c'est desormais mesure), elle
**arbitre les cas incertains** et interprete les ecarts.

Entree : sorties des Agents 2, 3, 4 et 5
Sortie : cartographie complete de comparaison (voir `process()`)
"""

import re
from collections import defaultdict

import numpy as np

from app.config import Config
from app.services import gemini_client, model_registry, reranking

# ---------------------------------------------------------------------------
# Contrat de caracteristiques
# ---------------------------------------------------------------------------
# Ordre FIGE : le classifieur entraine dans `notebooks/01_couverture.ipynb`
# doit produire ses colonnes dans exactement cet ordre. Toute evolution
# impose de reentrainer l'artefact.
CARACTERISTIQUES = [
    "cos_max",                 # meilleure similarite cosinus notion <-> unites
    "cos_moy_top3",            # moyenne des 3 meilleures similarites
    "cross_max",               # meilleur logit du cross-encodeur
    "cross_moy_top3",          # moyenne des 3 meilleurs logits
    "cross_prob_max",          # logit converti en pseudo-probabilite
    "recouvrement_lexical",    # part du vocabulaire de la notion present dans l'unite
    "nb_unites_fortes",        # nombre d'unites depassant le seuil de rappel
    "taille_unite_appariee",   # longueur (normalisee) de l'unite retenue
]

# Seuils de decision sur la probabilite de couverture.
SEUIL_COUVERTE = 0.62
SEUIL_PARTIELLE = 0.38
# Bande dans laquelle le verdict automatique n'est pas fiable : on le signale
# plutot que de trancher arbitrairement.
ZONE_INCERTITUDE = (0.42, 0.58)

# Calibration de la similarite cosinus, par moteur de vectorisation. Les
# echelles d'un Transformer et d'un espace LSA n'ont rien de comparable.
CALIBRATION_COSINUS = {
    "deep-learning": {"milieu": 0.50, "echelle": 0.09, "seuil_fort": 0.42},
    "machine-learning": {"milieu": 0.30, "echelle": 0.08, "seuil_fort": 0.20},
}

# Poids de fusion utilises tant que le classifieur entraine n'est pas depose.
# Le cross-encodeur domine largement : sur un corpus pedagogique, les
# similarites cosinus du bi-encodeur sont toutes elevees (tout est « des
# mathematiques »), ce qui en fait un signal peu discriminant. Le
# cross-encodeur, lui, separe nettement ce qui est enseigne de ce qui est
# seulement thematiquement proche.
POIDS_FUSION = {"cosinus": 0.25, "cross_encodeur": 0.75}

# Seuil de suffisance en deca duquel une notion pourtant abordee est jugee
# traitee trop brievement pour etre consideree comme reellement enseignee.
SEUIL_SUFFISANCE = 0.45
# Volume de texte (en mots, pondere par la pertinence) considere comme un
# traitement pedagogique complet d'une notion. En deca, le support releve
# davantage du plan de cours que de la lecon. Ordre de grandeur retenu :
# une lecon de primaire traitant reellement une notion (definition, exemple,
# application, exercice) represente au minimum ce volume.
MOTS_TRAITEMENT_COMPLET = 130.0

# Seuils d'origine, conserves comme dernier repli documente.
SEUILS_HISTORIQUES = {
    "deep-learning": {"couverte": 0.50, "partielle": 0.33},
    "machine-learning": {"couverte": 0.30, "partielle": 0.15},
}

_MOTS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}")
_VIDES = {
    "dans", "pour", "avec", "sans", "elle", "leur", "leurs", "cette", "cettes",
    "plus", "moins", "entre", "selon", "chaque", "tous", "toutes", "notion",
    "notions", "eleve", "eleves", "cours", "with", "from", "their", "which",
    "that", "this", "these", "those", "using", "such",
}


# ---------------------------------------------------------------------------
# Extraction des caracteristiques
# ---------------------------------------------------------------------------

def _mots_utiles(texte: str) -> set[str]:
    return {m for m in _MOTS.findall((texte or "").lower()) if m not in _VIDES}


def _recouvrement_lexical(notion: str, unite: str) -> float:
    """Part du vocabulaire significatif de la notion present dans l'unite.

    Signal volontairement naif, mais complementaire des deux modeles
    neuronaux : il capte les correspondances litterales (un terme technique
    repris a l'identique) que la similarite semantique peut sous-evaluer.
    """
    mots_notion = _mots_utiles(notion)
    if not mots_notion:
        return 0.0
    return len(mots_notion & _mots_utiles(unite)) / len(mots_notion)


def _construire_paires(agent3: dict, agent5: dict, top_k: int):
    """
    Prepare l'ensemble des paires (unite, notion) a soumettre au
    cross-encodeur, sans doublon, dans les deux sens d'interrogation.

    Retourne :
        textes_unites  : {unite_id: texte complet}
        paires         : [(texte_unite, texte_notion)]
        position       : {(unite_id, cle_notion): index dans `paires`}
    """
    textes_unites = {u["id"]: u["texte"] for u in agent3.get("unites", [])}

    paires: list[tuple[str, str]] = []
    position: dict[tuple[str, str], int] = {}

    def enregistrer(unite_id: str, cle_notion: str, texte_notion: str) -> None:
        if unite_id is None or (unite_id, cle_notion) in position:
            return
        texte_unite = textes_unites.get(unite_id, "")
        if not texte_unite:
            return
        position[(unite_id, cle_notion)] = len(paires)
        # Troncature : au-dela de ~900 caracteres le cross-encodeur tronque de
        # toute facon, autant economiser le temps de calcul.
        paires.append((texte_unite[:900], texte_notion[:400]))

    # Sens 1 : referentiel -> cours (decide de la couverture de chaque notion)
    for cle, entree in (agent5.get("meilleure_unite_par_notion") or {}).items():
        texte_notion = f"{entree['notion']}. {entree.get('descriptif', '')}".strip()
        for candidate in (entree.get("unites_candidates") or [])[:top_k]:
            enregistrer(candidate.get("unite_id"), cle, texte_notion)

    # Sens 2 : cours -> referentiel (detecte les contenus excedentaires).
    # Le meilleur voisin suffit : si meme lui est faible, l'unite est isolee.
    for voisinage in agent5.get("voisins_par_unite", []):
        voisins = voisinage.get("voisins") or []
        if not voisins:
            continue
        meilleur = voisins[0]
        cle = f"{meilleur['code']}::{meilleur['notion']}"
        enregistrer(voisinage["unite_id"], cle, meilleur["notion"])

    return textes_unites, paires, position


def _caracteristiques(entree: dict, textes_unites: dict, logits: np.ndarray | None,
                      position: dict, cle: str, seuil_fort: float) -> dict:
    """Calcule les huit caracteristiques d'une notion du referentiel."""
    candidates = (entree.get("unites_candidates") or [])[: Config.CROSS_ENCODER_TOP_K]
    cosinus = [float(c["score"]) for c in candidates] or [0.0]

    valeurs_cross: list[float] = []
    if logits is not None:
        for candidate in candidates:
            index = position.get((candidate.get("unite_id"), cle))
            if index is not None:
                valeurs_cross.append(float(logits[index]))

    # Volume de matiere reellement consacre a la notion : on additionne les
    # mots de chaque unite candidate, ponderes par leur pertinence. Une unite
    # longue mais hors sujet ne compte quasiment pas ; trois unites courtes
    # mais pertinentes se cumulent.
    mots_ponderes = 0.0
    for rang, candidate in enumerate(candidates):
        nb_mots = len(textes_unites.get(candidate.get("unite_id"), "").split())
        if rang < len(valeurs_cross):
            poids = reranking.probabilite_unique(valeurs_cross[rang])
        else:
            poids = float(np.clip((float(candidate["score"]) - 0.30) / 0.50, 0.0, 1.0))
        mots_ponderes += nb_mots * poids
    suffisance = float(np.clip(mots_ponderes / MOTS_TRAITEMENT_COMPLET, 0.0, 1.0))

    if valeurs_cross:
        ordre = sorted(valeurs_cross, reverse=True)
        cross_max = ordre[0]
        cross_moy = float(np.mean(ordre[:3]))
        cross_prob = reranking.probabilite_unique(cross_max)
        # L'unite retenue comme preuve est celle que le cross-encodeur juge la
        # plus pertinente, pas celle du bi-encodeur : c'est tout l'interet du
        # re-ranking.
        meilleur_index = int(np.argmax(valeurs_cross))
        unite_retenue = candidates[meilleur_index]
    else:
        # Imputation neutre : le classifieur doit voir une valeur coherente
        # avec « aucune information », pas un zero qui signifierait « nul ».
        cross_max = reranking.POINT_MILIEU
        cross_moy = reranking.POINT_MILIEU
        cross_prob = 0.5
        unite_retenue = candidates[0] if candidates else {}

    texte_unite = textes_unites.get(unite_retenue.get("unite_id"), "")
    texte_notion = f"{entree['notion']}. {entree.get('descriptif', '')}"

    return {
        "valeurs": {
            "cos_max": max(cosinus),
            "cos_moy_top3": float(np.mean(sorted(cosinus, reverse=True)[:3])),
            "cross_max": cross_max,
            "cross_moy_top3": cross_moy,
            "cross_prob_max": cross_prob,
            "recouvrement_lexical": _recouvrement_lexical(texte_notion, texte_unite),
            "nb_unites_fortes": float(sum(1 for c in cosinus if c >= seuil_fort)),
            "taille_unite_appariee": min(1.0, len(texte_unite.split()) / MOTS_TRAITEMENT_COMPLET),
        },
        "suffisance": suffisance,
        "unite_retenue": unite_retenue,
        "texte_unite": texte_unite,
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _fusion_logistique(valeurs: dict, calibration: dict, cross_actif: bool) -> float:
    """
    Repli employe tant que le classifieur calibre n'a pas ete depose.

    Chaque signal est ramene sur une echelle de probabilite par une fonction
    logistique, puis les deux sont combines. Ce n'est pas un modele appris :
    c'est une regle d'ingenierie explicite et documentee, remplacee des que
    l'artefact `couverture_clf.joblib` est disponible.
    """
    ecart = (valeurs["cos_max"] - calibration["milieu"]) / calibration["echelle"]
    p_cosinus = 1.0 / (1.0 + np.exp(-ecart))

    if not cross_actif:
        return float(np.clip(p_cosinus, 0.0, 1.0))

    p = (
        POIDS_FUSION["cosinus"] * p_cosinus
        + POIDS_FUSION["cross_encodeur"] * valeurs["cross_prob_max"]
    )
    # Le recouvrement litteral ne peut que confirmer, jamais infirmer : il
    # apporte un leger bonus quand le vocabulaire de la notion est repris tel
    # quel dans le cours.
    p += 0.08 * valeurs["recouvrement_lexical"] * (1.0 - p)
    return float(np.clip(p, 0.0, 1.0))


def _decider(toutes_valeurs: list[dict], calibration: dict, cross_actif: bool):
    """
    Retourne (probabilites, source_de_decision).

    Le classifieur entraine est prioritaire ; a defaut on applique la fusion
    logistique ci-dessus.
    """
    classifieur = model_registry.charger("couverture_clf")
    if classifieur is not None:
        try:
            matrice = np.array(
                [[v[nom] for nom in CARACTERISTIQUES] for v in toutes_valeurs],
                dtype="float64",
            )
            probabilites = classifieur.predict_proba(matrice)[:, 1]
            return np.clip(probabilites, 0.0, 1.0), "classifieur_calibre"
        except Exception:
            # Artefact incompatible (features renommees, version de sklearn
            # differente) : on retombe silencieusement sur la fusion.
            pass

    probabilites = np.array(
        [_fusion_logistique(v, calibration, cross_actif) for v in toutes_valeurs]
    )
    source = "fusion_logistique_cross_encodeur" if cross_actif else "fusion_logistique_cosinus"
    return probabilites, source


def _statut(probabilite: float) -> str:
    if probabilite >= SEUIL_COUVERTE:
        return "Couverte"
    if probabilite >= SEUIL_PARTIELLE:
        return "Partiellement couverte"
    return "Non couverte"


def _type_ecart(probabilite: float, suffisance: float, valeurs: dict,
                cross_actif: bool) -> str:
    """
    Qualifie la nature de l'ecart — bien plus actionnable qu'un simple
    « non couverte » : l'enseignant ne traite pas de la meme facon une notion
    absente, une notion evoquee sans etre enseignee, et une notion enseignee
    mais trop brievement.
    """
    if probabilite >= SEUIL_COUVERTE:
        return "traitee" if suffisance >= SEUIL_SUFFISANCE else "superficielle"

    # Le theme est reconnu par le bi-encodeur, mais le cross-encodeur ne voit
    # pas d'enseignement effectif de la notion : c'est le cas typique du
    # programme qui « parle de geometrie » sans traiter la notion visee.
    theme_present = valeurs["cos_max"] >= 0.52
    enseignement_faible = (not cross_actif) or valeurs["cross_prob_max"] < 0.45

    if theme_present and enseignement_faible:
        return "evoquee_non_enseignee"
    if probabilite >= SEUIL_PARTIELLE:
        return "amorcee"
    return "absente"


LIBELLES_ECART = {
    "traitee": "Traitée",
    "superficielle": "Traitée trop brièvement",
    "evoquee_non_enseignee": "Évoquée mais non enseignée",
    "amorcee": "Amorcée, insuffisante",
    "absente": "Absente du support",
}


# ---------------------------------------------------------------------------
# Lecture qualitative (modele de langage)
# ---------------------------------------------------------------------------

PROMPT = """Tu es un expert en ingenierie pedagogique et en accreditation internationale.

Un systeme automatise a compare un cours de {matiere} (niveau « {niveau} », systeme
educatif marocain) a plusieurs referentiels etrangers. La couverture de chaque notion
a ete MESUREE par un bi-encodeur semantique puis un cross-encodeur de re-ranking : les
chiffres ci-dessous sont des mesures, pas des estimations a revoir.

Titre du cours : {titre_cours}
Resume : {resume}
Chapitres : {chapitres}

Couverture mesuree par referentiel :
{synthese_chiffree}

Notions insuffisamment couvertes (avec la nature de l'ecart diagnostiquee) :
{notions_faibles}

Notions abordees mais traitees TROP BRIEVEMENT pour etre reellement apprises
(le support les mentionne, sans y consacrer assez de matiere) :
{notions_superficielles}

Notions situees dans la ZONE D'INCERTITUDE du modele (probabilite entre 0,42 et 0,58 —
le systeme ne sait pas trancher) :
{notions_incertaines}

Contenus du cours sans equivalent dans les referentiels compares :
{contenus_excedentaires}

Ta mission comporte deux volets :
A) INTERPRETER ces resultats en pedagogue (ne repete pas les chiffres).
B) ARBITRER les notions de la zone d'incertitude : pour chacune, dis si tu la
   considères plutot couverte ou plutot manquante, et pourquoi.

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour :
{{
  "synthese_comparative": "3 a 4 phrases interpretant le positionnement du cours",
  "analyse_par_pays": [
    {{"pays": "nom du pays", "lecture": "1 a 2 phrases sur les specificites de ce referentiel face au cours"}}
  ],
  "arbitrages_incertains": [
    {{"notion": "intitule exact", "verdict": "plutot_couverte" | "plutot_manquante", "justification": "une phrase"}}
  ],
  "ecarts_majeurs": [
    {{"notion": "notion manquante", "pays": "referentiel concerne", "impact": "consequence pedagogique concrete"}}
  ],
  "points_forts": ["point fort du cours", "..."],
  "specificites_locales": ["contenu propre au cours marocain a valoriser", "..."]
}}

Contraintes : francais, formulations courtes et concretes, maximum 5 elements par liste.
"""


def _texte_synthese_chiffree(par_pays: dict) -> str:
    lignes = []
    for donnees in par_pays.values():
        lignes.append(
            f"- {donnees['pays']} ({donnees['referentiel']}) : couverture {donnees['taux_couverture_pct']} % "
            f"({donnees['nb_couvertes']} couvertes dont {donnees['nb_approfondies']} traitees en profondeur, "
            f"{donnees['nb_partielles']} partielles, {donnees['nb_manquantes']} manquantes "
            f"sur {donnees['nb_notions']})"
        )
    return "\n".join(lignes) or "(aucun referentiel selectionne)"


def _analyse_gemini(agent2: dict, par_pays: dict, notions_faibles: list[dict],
                    notions_incertaines: list[dict], notions_superficielles: list[dict],
                    excedentaires: list[dict], matiere: str, niveau: str) -> dict:
    superficielles = "\n".join(
        f"- [{n['pays']}] {n['notion']} (suffisance {n['suffisance']} — "
        f"chapitre « {n['chapitre_correspondant']} »)"
        for n in notions_superficielles[:15]
    ) or "(aucune)"
    faibles = "\n".join(
        f"- [{n['pays']}] {n['notion']} — {LIBELLES_ECART.get(n['type_ecart'], n['type_ecart'])} "
        f"(probabilite de couverture {n['probabilite_couverture']})"
        for n in notions_faibles[:25]
    ) or "(aucune)"
    incertaines = "\n".join(
        f"- [{n['pays']}] {n['notion']} (probabilite {n['probabilite_couverture']}) — "
        f"extrait du cours le plus proche : « {n['extrait_correspondant'][:140]} »"
        for n in notions_incertaines[:12]
    ) or "(aucune)"
    surplus = "\n".join(
        f"- {c['chapitre']} : {c['extrait'][:120]}" for c in excedentaires[:10]
    ) or "(aucun)"
    chapitres = ", ".join(c["titre"] for c in agent2.get("chapitres", [])[:12]) or "(aucun)"

    prompt = PROMPT.format(
        matiere=matiere,
        niveau=niveau,
        titre_cours=agent2.get("titre_cours", ""),
        resume=agent2.get("resume", ""),
        chapitres=chapitres,
        synthese_chiffree=_texte_synthese_chiffree(par_pays),
        notions_faibles=faibles,
        notions_superficielles=superficielles,
        notions_incertaines=incertaines,
        contenus_excedentaires=surplus,
    )

    try:
        donnees = gemini_client.generate_json(prompt, agent="agent6_comparaison")
        if not isinstance(donnees, dict):
            raise ValueError("structure JSON inattendue")
        donnees["disponible"] = True
        donnees["erreur"] = None
        donnees.setdefault("arbitrages_incertains", [])
        return donnees
    except Exception as exc:
        return {
            "disponible": False,
            "erreur": str(exc),
            "synthese_comparative": "",
            "analyse_par_pays": [],
            "arbitrages_incertains": [],
            "ecarts_majeurs": [],
            "points_forts": [],
            "specificites_locales": [],
        }


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def process(agent2: dict, agent3: dict, agent4: dict, agent5: dict,
            matiere: str, niveau: str) -> dict:
    """
    Execute l'Agent 6.

    Retourne la cartographie complete : `par_pays`, `notions_communes`,
    `notions_manquantes`, `contenus_excedentaires`, `score_global_pct`,
    la tracabilite du moteur de decision et l'analyse qualitative `gemini`.
    """
    type_moteur = agent4.get("type_moteur", "deep-learning")
    calibration = CALIBRATION_COSINUS.get(type_moteur, CALIBRATION_COSINUS["deep-learning"])

    # --- Etage 2 : re-ranking par cross-encodeur ------------------------
    textes_unites, paires, position = _construire_paires(
        agent3, agent5, Config.CROSS_ENCODER_TOP_K
    )
    logits = reranking.scorer_paires(paires) if paires else None
    cross_actif = logits is not None and len(logits) == len(paires)
    if not cross_actif:
        logits = None

    # --- Etage 3 : caracteristiques puis decision -----------------------
    entrees = list((agent5.get("meilleure_unite_par_notion") or {}).items())
    extraits = [
        _caracteristiques(entree, textes_unites, logits, position, cle,
                          calibration["seuil_fort"])
        for cle, entree in entrees
    ]
    probabilites, source_decision = (
        _decider([e["valeurs"] for e in extraits], calibration, cross_actif)
        if extraits
        else (np.zeros(0), "aucune_notion")
    )

    # --- Cartographie par pays ------------------------------------------
    par_pays: dict[str, dict] = {}
    detail_par_pays: dict[str, list] = defaultdict(list)
    notions_manquantes: list[dict] = []
    notions_faibles: list[dict] = []
    notions_incertaines: list[dict] = []
    statut_par_intitule: dict[str, list[str]] = defaultdict(list)

    for (cle, entree), extrait, probabilite in zip(entrees, extraits, probabilites):
        probabilite = float(probabilite)
        valeurs = extrait["valeurs"]
        statut = _statut(probabilite)
        suffisance = extrait["suffisance"]
        incertaine = ZONE_INCERTITUDE[0] <= probabilite <= ZONE_INCERTITUDE[1]
        unite = extrait["unite_retenue"]

        ligne = {
            "notion": entree["notion"],
            "descriptif": entree.get("descriptif", ""),
            "pays": entree["pays"],
            "code": entree["code"],
            # `score` reste la similarite cosinus brute, conservee pour la
            # continuite d'affichage et la comparaison avec l'ancienne version.
            "score": round(valeurs["cos_max"], 3),
            "probabilite_couverture": round(probabilite, 3),
            "suffisance": round(suffisance, 3),
            "statut": statut,
            "type_ecart": _type_ecart(probabilite, suffisance, valeurs, cross_actif),
            "incertaine": incertaine,
            "traitement_approfondi": bool(
                statut == "Couverte" and suffisance >= SEUIL_SUFFISANCE
            ),
            "chapitre_correspondant": unite.get("chapitre"),
            "page_correspondante": unite.get("page"),
            "extrait_correspondant": (extrait["texte_unite"] or "")[:260],
            "caracteristiques": {k: round(float(v), 4) for k, v in valeurs.items()},
        }
        ligne["libelle_ecart"] = LIBELLES_ECART.get(ligne["type_ecart"], ligne["type_ecart"])

        detail_par_pays[entree["code"]].append(ligne)
        statut_par_intitule[entree["notion"].strip().lower()].append(statut)
        if statut == "Non couverte":
            notions_manquantes.append(ligne)
        if statut != "Couverte":
            notions_faibles.append(ligne)
        if incertaine:
            notions_incertaines.append(ligne)

    source_notions = agent5.get("couverture_brute_par_pays") or {}
    for code, detail in detail_par_pays.items():
        detail.sort(key=lambda n: n["probabilite_couverture"], reverse=True)
        nb = len(detail) or 1
        nb_couvertes = sum(1 for n in detail if n["statut"] == "Couverte")
        nb_partielles = sum(1 for n in detail if n["statut"] == "Partiellement couverte")
        nb_manquantes = sum(1 for n in detail if n["statut"] == "Non couverte")
        origine = source_notions.get(code) or [{}]

        par_pays[code] = {
            "code": code,
            "pays": detail[0]["pays"] if detail else code,
            "drapeau": next((n.get("drapeau", "") for n in origine if n.get("drapeau")), ""),
            "referentiel": next((n.get("referentiel", "") for n in origine if n.get("referentiel")), ""),
            "notions": detail,
            "nb_notions": len(detail),
            "nb_couvertes": nb_couvertes,
            "nb_partielles": nb_partielles,
            "nb_manquantes": nb_manquantes,
            "nb_incertaines": sum(1 for n in detail if n["incertaine"]),
            "nb_approfondies": sum(1 for n in detail if n["traitement_approfondi"]),
            "nb_superficielles": sum(1 for n in detail if n["type_ecart"] == "superficielle"),
            # Taux discret : une notion couverte compte pour 1, une notion
            # partiellement couverte pour 0,5. Lisible par un non-specialiste.
            "taux_couverture_pct": round(100 * (nb_couvertes + 0.5 * nb_partielles) / nb, 1),
            # Taux continu : moyenne des probabilites. Plus fidele, sans effet
            # de seuil, mais moins intuitif — les deux sont restitues.
            "taux_couverture_probabiliste_pct": round(
                100 * float(np.mean([n["probabilite_couverture"] for n in detail])), 1
            ),
            # Part des notions a la fois couvertes ET traitees avec assez de
            # matiere : c'est l'indicateur qui distingue un plan de cours d'un
            # support d'enseignement reellement exploitable.
            "taux_traitement_approfondi_pct": round(
                100 * sum(1 for n in detail if n["traitement_approfondi"]) / nb, 1
            ),
            "score_similarite_moyen": round(
                float(np.mean([n["score"] for n in detail])), 3
            ),
        }

    # --- Notions communes a tous les referentiels selectionnes ----------
    nb_pays = len(par_pays) or 1
    notions_communes = sorted(
        intitule
        for intitule, statuts in statut_par_intitule.items()
        if len(statuts) >= nb_pays and all(s != "Non couverte" for s in statuts)
    )

    # --- Contenus excedentaires ------------------------------------------
    # Une unite est excedentaire si meme sa meilleure correspondance ne resiste
    # pas au re-ranking : le theme peut sembler proche, l'enseignement ne l'est pas.
    contenus_excedentaires = []
    for voisinage in agent5.get("voisins_par_unite", []):
        voisins = voisinage.get("voisins") or []
        if not voisins:
            continue
        meilleur = voisins[0]
        cos_max = max((float(v["score"]) for v in voisins), default=0.0)

        index = position.get(
            (voisinage["unite_id"], f"{meilleur['code']}::{meilleur['notion']}")
        )
        if cross_actif and index is not None:
            probabilite = reranking.probabilite_unique(float(logits[index]))
            isole = probabilite < 0.35
        else:
            probabilite = None
            isole = cos_max < calibration["milieu"] - 2 * calibration["echelle"]

        if isole:
            contenus_excedentaires.append({
                "unite_id": voisinage["unite_id"],
                "chapitre": voisinage["chapitre"],
                "page": voisinage.get("page"),
                "extrait": voisinage["extrait"],
                "meilleur_score": round(cos_max, 3),
                "meilleure_probabilite": round(probabilite, 3) if probabilite is not None else None,
                "notion_la_plus_proche": meilleur["notion"],
            })

    # --- Score global -----------------------------------------------------
    score_global = (
        round(float(np.mean([p["taux_couverture_pct"] for p in par_pays.values()])), 1)
        if par_pays else 0.0
    )
    score_global_probabiliste = (
        round(float(np.mean([p["taux_couverture_probabiliste_pct"] for p in par_pays.values()])), 1)
        if par_pays else 0.0
    )
    score_approfondissement = (
        round(float(np.mean([p["taux_traitement_approfondi_pct"] for p in par_pays.values()])), 1)
        if par_pays else 0.0
    )
    notions_superficielles = [
        n for detail in detail_par_pays.values() for n in detail
        if n["type_ecart"] == "superficielle"
    ]

    # --- Lecture qualitative ---------------------------------------------
    notions_manquantes.sort(key=lambda n: n["probabilite_couverture"])
    analyse = _analyse_gemini(
        agent2, par_pays, notions_faibles, notions_incertaines,
        notions_superficielles, contenus_excedentaires, matiere, niveau,
    )

    return {
        "matiere": matiere,
        "niveau": niveau,
        "moteur_similarite": agent4.get("moteur"),
        "type_moteur": type_moteur,
        "reranking": {
            **reranking.infos(),
            "applique": cross_actif,
            "nb_paires_scorees": len(paires) if cross_actif else 0,
        },
        "decision": {
            "source": source_decision,
            "caracteristiques": CARACTERISTIQUES,
            "seuils": {"couverte": SEUIL_COUVERTE, "partielle": SEUIL_PARTIELLE},
            "zone_incertitude": list(ZONE_INCERTITUDE),
            "calibration_cosinus": calibration,
            "seuils_historiques": SEUILS_HISTORIQUES.get(type_moteur, {}),
        },
        # Conserve pour la compatibilite du rapport et de l'export PDF.
        "seuils_appliques": SEUILS_HISTORIQUES.get(type_moteur, SEUILS_HISTORIQUES["deep-learning"]),
        "par_pays": par_pays,
        "score_global_pct": score_global,
        "score_global_probabiliste_pct": score_global_probabiliste,
        "score_approfondissement_pct": score_approfondissement,
        "notions_superficielles": notions_superficielles,
        "nb_notions_superficielles": len(notions_superficielles),
        "notions_communes": notions_communes,
        "notions_manquantes": notions_manquantes,
        "nb_notions_manquantes": len(notions_manquantes),
        "notions_incertaines": notions_incertaines,
        "nb_notions_incertaines": len(notions_incertaines),
        "contenus_excedentaires": contenus_excedentaires,
        "nb_contenus_excedentaires": len(contenus_excedentaires),
        "gemini": analyse,
    }
