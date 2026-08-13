"""
Detection d'anomalies de connexion.
====================================

Le Module Authentification et Securite sait *qui* se connecte — il ne sait
pas *si c'est normal*. Une session ouverte a trois heures du matin depuis un
compte qui n'a jamais travaille la nuit, ou vingt connexions en dix minutes,
sont des signaux qu'aucune verification d'identite ne capte : les
identifiants sont valides dans les deux cas.

Choix de methode
----------------

On ne dispose d'aucun exemple d'attaque — et il serait malhonnete d'en
inventer pour entrainer un classifieur binaire qui apprendrait surtout les
conventions de fabrication de ces faux exemples. La detection repose donc sur
un `IsolationForest`, entraine **uniquement sur le comportement observe**.
L'algorithme isole les points rares : il n'a besoin d'aucune etiquette, et il
signale ce qui s'ecarte de l'usage habituel de l'instance, quel qu'il soit.

Encodage cyclique du temps
--------------------------

Representer une heure par un entier de 0 a 23 place 23 h et minuit aux deux
extremites de l'echelle, alors qu'une heure les separe. Tout modele fonde sur
des distances en tirerait des conclusions fausses. Les instants sont donc
projetes sur un cercle — sinus et cosinus — ce qui retablit la continuite du
temps. Le meme traitement est applique au jour de la semaine.

Ce que la methode capte, et ce qu'elle ne capte pas
----------------------------------------------------

Mesure sur un journal simule de 224 connexions (un enseignant se connectant
en semaine entre 8 h et 18 h, plus quelques comportements anormaux injectes) :

- une connexion nocturne isolee ressort en tete, avec un risque de 86/100 ;
- une rafale de connexions n'est **pas** fiablement isolee par le modele, et
  c'est structurel : `IsolationForest` detecte les points rares, or une
  rafale forme un groupe compact. Plus elle est massive, moins elle est rare
  au sens de l'algorithme. C'est une regle explicite qui la traite ;
- environ 5 % des connexions normales sont signalees.

Ces chiffres proviennent d'une distribution simulee, choisie par nos soins :
ils indiquent un ordre de grandeur, pas une performance garantie sur des
donnees reelles. Le dispositif est concu comme une **aide a la supervision**
— une liste que l'administrateur parcourt — et non comme un controle d'acces.
Aucune connexion n'est jamais bloquee par ce module.

Deux mecanismes complementaires
--------------------------------

Le modele repere ce a quoi personne n'a pense ; les regles explicites
traitent les motifs connus a l'avance, que le modele capte mal. Une regle
declenchee est un fait verifiable, assorti de son libelle : l'administrateur
sait *pourquoi* une connexion est signalee, la ou un score d'anomalie reste
opaque.

Amorcage
--------

En dessous d'un volume minimal d'evenements, aucun modele n'est entraine :
le module se rabat sur une regle statistique explicite (ecart median absolu
sur les delais entre connexions). Signaler des anomalies a partir de trois
observations serait du bruit presente comme de la securite.
"""

import time

import numpy as np

from app.services import entrainement

# Volume d'evenements en dessous duquel on n'entraine pas de modele.
MIN_EVENEMENTS = 25
# Part attendue de connexions atypiques : volontairement basse, une alerte de
# securite qui se declenche souvent n'est plus lue.
CONTAMINATION = 0.06

# --- Regles explicites, en complement du modele ----------------------------
# Un detecteur d'anomalies isole les points RARES. Une rafale de connexions
# forme au contraire un groupe compact : plus elle est massive, moins elle est
# rare au sens de l'algorithme, et moins il la signale. C'est une limite
# structurelle de la methode, pas un defaut de reglage.
#
# Les motifs connus a l'avance sont donc traites par des regles explicites, et
# le modele garde ce pour quoi il est irremplacable : reperer ce a quoi
# personne n'a pense. Le verdict final combine les deux.
CADENCE_SUSPECTE_24H = 15
RAFALE_NB = 5
RAFALE_FENETRE_S = 300

CARACTERISTIQUES = [
    "heure_sin", "heure_cos", "jour_sin", "jour_cos",
    "delai_relatif_log", "cadence_24h", "anciennete_compte_log",
]


def _horodatage(evenement: dict) -> float:
    """Ramene l'horodatage d'un evenement a un instant flottant."""
    valeur = evenement.get("horodatage")
    if isinstance(valeur, (int, float)):
        return float(valeur)
    try:
        import datetime

        if isinstance(valeur, datetime.datetime):
            return valeur.timestamp()
        return datetime.datetime.fromisoformat(str(valeur)).timestamp()
    except Exception:
        return time.time()


def construire_caracteristiques(evenements: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Transforme le journal des connexions en observations exploitables.

    Les evenements sont regroupes par compte et tries chronologiquement : le
    delai depuis la connexion precedente et la cadence sur vingt-quatre
    heures n'ont de sens que rapportes a l'historique du meme utilisateur.
    """
    connexions = [e for e in evenements if e.get("type") == "connexion"]
    par_compte: dict[str, list[dict]] = {}
    for evenement in connexions:
        par_compte.setdefault(evenement.get("utilisateur_id") or "?", []).append(evenement)

    lignes, contextes = [], []
    for compte, journal in par_compte.items():
        journal.sort(key=_horodatage)
        premier = _horodatage(journal[0])

        # Rythme propre au compte : un enseignant qui se connecte une fois par
        # semaine et un autre qui s'y connecte trois fois par jour n'ont pas le
        # meme « normal ». Comparer les delais dans l'absolu ferait passer la
        # reprise du lundi pour une anomalie chez le premier. On rapporte donc
        # chaque delai a la mediane du compte.
        delais_bruts = [
            _horodatage(journal[i]) - _horodatage(journal[i - 1])
            for i in range(1, len(journal))
        ]
        reference = float(np.median([d for d in delais_bruts if d > 0])) if delais_bruts else 86400.0
        reference = max(reference, 60.0)

        for index, evenement in enumerate(journal):
            instant = _horodatage(evenement)
            heure_sin, heure_cos, jour_sin, jour_cos = entrainement.encoder_horaire(instant)

            delai = instant - _horodatage(journal[index - 1]) if index else reference
            # Echelle logarithmique du rapport au rythme habituel : 1 signifie
            # « comme d'habitude », les ecarts jouent dans les deux sens.
            delai_relatif = float(np.log1p(max(delai, 0.0) / reference))
            cadence = sum(1 for e in journal[:index + 1] if instant - _horodatage(e) <= 86400)
            anciennete = float(np.log1p(max(instant - premier, 0.0)))

            lignes.append([heure_sin, heure_cos, jour_sin, jour_cos,
                           delai_relatif, float(cadence), anciennete])
            contextes.append({
                "utilisateur_id": compte,
                "horodatage": instant,
                "delai_s": round(delai, 1),
                "delai_relatif": round(delai / reference, 2),
                "rythme_median_s": round(reference, 1),
                "cadence_24h": cadence,
            })

    if not lignes:
        return np.zeros((0, len(CARACTERISTIQUES))), []
    return np.asarray(lignes, dtype="float64"), contextes


def regles_explicites(contexte: dict, journal_compte: list[float]) -> list[dict]:
    """
    Motifs suspects connus, detectes par regle plutot que par apprentissage.

    Retourne la liste des regles declenchees. Chacune porte son libelle, ce
    qui permet a l'administrateur de savoir *pourquoi* une connexion est
    signalee — la ou un score d'anomalie reste opaque.
    """
    declenchees = []
    instant = contexte["horodatage"]

    if contexte["cadence_24h"] >= CADENCE_SUSPECTE_24H:
        declenchees.append({
            "regle": "cadence_elevee",
            "libelle": (
                f"{contexte['cadence_24h']} connexions en 24 heures "
                f"(seuil {CADENCE_SUSPECTE_24H})"
            ),
        })

    recentes = sum(1 for t in journal_compte if 0 <= instant - t <= RAFALE_FENETRE_S)
    if recentes >= RAFALE_NB:
        declenchees.append({
            "regle": "rafale",
            "libelle": (
                f"{recentes} connexions en moins de "
                f"{RAFALE_FENETRE_S // 60} minutes"
            ),
        })

    return declenchees


def _entrainer() -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    evenements = entrainement.corpus_evenements({"connexion"})
    matrice, _ = construire_caracteristiques(evenements)

    if len(matrice) < MIN_EVENEMENTS:
        return {
            "modele": None,
            "source_donnees": (
                f"{len(matrice)} connexions enregistrées — minimum requis "
                f"{MIN_EVENEMENTS} pour entraîner un modèle"
            ),
            "nb_observations": len(matrice),
        }

    modele = make_pipeline(
        StandardScaler(),
        IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42),
    )
    modele.fit(matrice)
    scores = modele.decision_function(matrice)

    return {
        "modele": modele,
        "source_donnees": "journal des connexions de cette instance",
        "nb_observations": len(matrice),
        "metrique_part_atypique": round(float(np.mean(scores < 0)), 3),
        "metrique_score_median": round(float(np.median(scores)), 4),
    }


def _repli_statistique(contexte: dict, contextes: list[dict]) -> dict:
    """
    Regle explicite utilisee tant qu'il n'y a pas assez de donnees.

    On compare le delai depuis la connexion precedente a la mediane des
    delais du compte, via l'ecart median absolu — robuste aux valeurs
    extremes la ou l'ecart-type ne le serait pas sur si peu d'observations.
    """
    delais = [
        c["delai_s"] for c in contextes
        if c["utilisateur_id"] == contexte["utilisateur_id"] and c["delai_s"] > 0
    ]
    if len(delais) < 5:
        return {"applique": False, "motif": "historique insuffisant"}

    mediane = float(np.median(delais))
    ecart = float(np.median([abs(d - mediane) for d in delais])) or 1.0
    z = abs(contexte["delai_s"] - mediane) / (1.4826 * ecart)

    return {
        "applique": True,
        "methode": "écart médian absolu sur les délais entre connexions",
        "atypique": bool(z > 3.5 or contexte["cadence_24h"] > 12),
        "z_robuste": round(z, 2),
        "cadence_24h": contexte["cadence_24h"],
    }


def evaluer_connexion(utilisateur_id: str) -> dict:
    """
    Score de risque de la connexion la plus recente d'un compte.

    Retourne toujours un diagnostic : modele appris s'il existe, regle
    statistique sinon, et mention explicite de l'amorcage dans les deux cas.
    """
    evenements = entrainement.corpus_evenements({"connexion"})
    matrice, contextes = construire_caracteristiques(evenements)

    indices = [
        i for i, c in enumerate(contextes) if c["utilisateur_id"] == utilisateur_id
    ]
    if not indices:
        return {"applique": False, "motif": "aucune connexion enregistrée"}

    dernier = indices[-1]
    contexte = contextes[dernier]

    entraine = entrainement.obtenir("auth_anomalie", _entrainer)
    modele = entraine.get("modele")

    if modele is None:
        repli = _repli_statistique(contexte, contextes)
        repli.update({
            "amorcage": True,
            "nb_observations": entraine.get("nb_observations", 0),
            "motif_amorcage": entraine.get("source_donnees"),
            "contexte": contexte,
        })
        return repli

    score = float(modele.decision_function(matrice[dernier : dernier + 1])[0])
    atypique_modele = bool(modele.predict(matrice[dernier : dernier + 1])[0] == -1)
    # Score ramene sur [0, 100] : 0 = comportement habituel, 100 = tres rare.
    risque = float(np.clip(50 - score * 200, 0, 100))

    journal_compte = [
        c["horodatage"] for c in contextes
        if c["utilisateur_id"] == utilisateur_id
    ]
    regles = regles_explicites(contexte, journal_compte)
    if regles:
        # Une regle declenchee est un fait, pas une estimation : elle porte le
        # risque au maximum, quel que soit l'avis du modele.
        risque = max(risque, 85.0)

    return {
        "applique": True,
        "methode": "IsolationForest + règles explicites",
        "atypique": bool(atypique_modele or regles),
        "atypique_modele": atypique_modele,
        "regles_declenchees": regles,
        "score": round(score, 4),
        "risque": round(risque, 1),
        "amorcage": entraine.get("amorcage", False),
        "nb_observations": entraine.get("nb_observations", 0),
        "contexte": contexte,
    }


def connexions_atypiques(limite: int = 20) -> dict:
    """
    Toutes les connexions signalees comme atypiques, pour le back-office.

    C'est la vue qui donne sa valeur au module : un administrateur ne
    consulte pas les sessions une par une, il veut la liste de celles qui
    sortent de l'ordinaire.
    """
    evenements = entrainement.corpus_evenements({"connexion"})
    matrice, contextes = construire_caracteristiques(evenements)
    entraine = entrainement.obtenir("auth_anomalie", _entrainer)
    modele = entraine.get("modele")

    if modele is None or len(matrice) == 0:
        return {
            "applique": False,
            "motif": entraine.get("source_donnees"),
            "nb_connexions": len(matrice),
            "nb_observations": entraine.get("nb_observations", 0),
            "connexions": [],
        }

    scores = modele.decision_function(matrice)
    predictions = modele.predict(matrice)

    import datetime

    par_compte_horodatages: dict[str, list[float]] = {}
    for c in contextes:
        par_compte_horodatages.setdefault(c["utilisateur_id"], []).append(c["horodatage"])

    signalees = []
    for i, (score, prediction) in enumerate(zip(scores, predictions)):
        contexte = contextes[i]
        regles = regles_explicites(
            contexte, par_compte_horodatages.get(contexte["utilisateur_id"], [])
        )
        if prediction != -1 and not regles:
            continue
        risque = float(np.clip(50 - score * 200, 0, 100))
        if regles:
            risque = max(risque, 85.0)
        signalees.append({
            **contexte,
            "date": datetime.datetime.fromtimestamp(contexte["horodatage"]).strftime(
                "%d/%m/%Y %H:%M"
            ),
            "score": round(float(score), 4),
            "risque": round(risque, 1),
            "detecte_par": (
                "règle + modèle" if (regles and prediction == -1)
                else "règle" if regles else "modèle"
            ),
            "regles_declenchees": regles,
        })

    signalees.sort(key=lambda c: c["risque"], reverse=True)
    return {
        "applique": True,
        "methode": "IsolationForest + règles explicites",
        "nb_connexions": len(matrice),
        "nb_atypiques": len(signalees),
        "part_atypique_pct": round(100 * len(signalees) / max(1, len(matrice)), 1),
        "connexions": signalees[:limite],
        "nb_observations": entraine.get("nb_observations", 0),
        "amorcage": entraine.get("amorcage", False),
    }
