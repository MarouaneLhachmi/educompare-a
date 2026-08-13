"""
Module Supervision des Modeles  (sixieme module fonctionnel)
=============================================================

Pourquoi ce module existe
--------------------------

Le systeme compte desormais une douzaine de modeles : cross-encodeur,
ensemble de regression, planificateur par renforcement, triage documentaire,
prediction de duree, detection d'anomalies, regroupement des analyses. Les
cinq modules d'origine n'avaient rien a superviser ; ce n'est plus le cas.

Un modele ne tombe pas en panne : il **se degrade silencieusement**. Le
triage documentaire a ete entraine sur des cours de mathematiques du
primaire ; le jour ou l'etablissement depose des supports de terminale, ses
predictions restent syntaxiquement valides et deviennent pedagogiquement
fausses. Rien dans les journaux applicatifs ne le signale.

C'est la fonction de ce module : mesurer l'ecart entre les donnees sur
lesquelles un modele a appris et celles qu'il traite aujourd'hui, et
declencher son reentrainement avant que ses sorties ne soient plus fiables.

Deux mesures de derive
-----------------------

**Population Stability Index.** On decoupe la distribution de reference en
deciles, on compte la part de la population courante tombant dans chaque
tranche, et on somme

    PSI = somme (part_courante - part_reference) x ln(part_courante / part_reference)

Les seuils sont ceux en usage dans l'industrie du credit, ou l'indicateur
est ne : en dessous de 0,10 la population est stable, entre 0,10 et 0,25 la
derive est moderee, au-dela elle est significative.

**Test de Kolmogorov-Smirnov.** Le PSI depend du decoupage retenu ; le test
de KS n'en depend pas. Il compare les fonctions de repartition empiriques et
retourne une p-valeur. Les deux mesures sont donnees ensemble : quand elles
divergent, c'est que la derive porte sur la forme de la distribution plutot
que sur sa position.

Ce que le module ne fait pas
-----------------------------

Il ne juge pas de la *qualite* des predictions — cela exigerait des
etiquettes que le systeme n'a pas. Il mesure la derive des **entrees**, ce
qui est un signal precoce et non une preuve de degradation. La distinction
est explicite dans les libelles restitues a l'administrateur.
"""

import time

import numpy as np

from app.services import entrainement, model_registry

# Seuils usuels du Population Stability Index.
PSI_STABLE = 0.10
PSI_MODERE = 0.25
# Une p-valeur de KS en dessous de ce seuil rejette l'hypothese de
# distributions identiques.
SEUIL_KS = 0.05
# Age au-dela duquel un modele merite d'etre reentraine, meme sans derive :
# le corpus a pu s'enrichir sans que sa distribution change.
AGE_MAX_HEURES = 72
# Nouvelles analyses accumulees au-dela desquelles un reentrainement se
# justifie, la matiere d'apprentissage ayant sensiblement grossi.
NOUVELLES_ANALYSES_SEUIL = 10


# ---------------------------------------------------------------------------
# Mesures de derive
# ---------------------------------------------------------------------------

def psi(reference: np.ndarray, courant: np.ndarray, nb_tranches: int = 10) -> float:
    """
    Population Stability Index entre deux echantillons.

    Les bornes des tranches viennent de la distribution de **reference** :
    c'est elle qui definit le decoupage, la population courante etant ensuite
    projetee dessus. Un lissage evite la division par zero quand une tranche
    se vide completement.
    """
    reference = np.asarray(reference, dtype="float64")
    courant = np.asarray(courant, dtype="float64")
    if len(reference) < 10 or len(courant) < 5:
        return float("nan")

    quantiles = np.linspace(0, 100, nb_tranches + 1)
    bornes = np.unique(np.percentile(reference, quantiles))
    if len(bornes) < 3:
        return 0.0
    bornes[0], bornes[-1] = -np.inf, np.inf

    part_reference = np.histogram(reference, bins=bornes)[0] / len(reference)
    part_courant = np.histogram(courant, bins=bornes)[0] / len(courant)

    lissage = 1e-4
    part_reference = np.clip(part_reference, lissage, None)
    part_courant = np.clip(part_courant, lissage, None)

    return float(np.sum((part_courant - part_reference)
                        * np.log(part_courant / part_reference)))


def kolmogorov_smirnov(reference: np.ndarray, courant: np.ndarray) -> dict:
    """Test de KS a deux echantillons."""
    from scipy import stats

    reference = np.asarray(reference, dtype="float64")
    courant = np.asarray(courant, dtype="float64")
    if len(reference) < 5 or len(courant) < 5:
        return {"applique": False, "motif": "échantillons trop petits"}

    resultat = stats.ks_2samp(reference, courant)
    return {
        "applique": True,
        "statistique": round(float(resultat.statistic), 4),
        "p_valeur": round(float(resultat.pvalue), 4),
        "distributions_differentes": bool(resultat.pvalue < SEUIL_KS),
    }


def _qualifier(valeur_psi: float) -> tuple[str, str]:
    if np.isnan(valeur_psi):
        return "indeterminee", "Échantillon insuffisant pour conclure."
    if valeur_psi < PSI_STABLE:
        return "stable", "La population traitée ressemble à celle de l'entraînement."
    if valeur_psi < PSI_MODERE:
        return "moderee", "Dérive modérée : à surveiller, sans urgence."
    return "significative", "Dérive significative : le modèle doit être réentraîné."


# ---------------------------------------------------------------------------
# Sources de donnees surveillees
# ---------------------------------------------------------------------------

def _distributions_analyses() -> dict:
    """
    Caracteristiques des documents analyses, decoupees en une periode de
    reference (la plus ancienne moitie) et une periode courante (la plus
    recente). C'est le decoupage le plus honnete en l'absence d'un instantane
    fige au moment de l'entrainement.
    """
    analyses = sorted(
        entrainement.corpus_analyses(),
        key=lambda a: str(a.get("date_creation_iso") or ""),
    )
    if len(analyses) < 12:
        return {"applique": False, "nb_analyses": len(analyses)}

    milieu = len(analyses) // 2
    ancien, recent = analyses[:milieu], analyses[milieu:]

    def extraire(lot, cle):
        valeurs = []
        for analyse in lot:
            if cle == "nb_mots":
                valeur = (analyse.get("agent1") or {}).get("nb_mots")
            elif cle == "nb_unites":
                valeur = (analyse.get("agent3") or {}).get("nb_unites")
            elif cle == "note":
                valeur = analyse.get("resume_note_globale")
            elif cle == "couverture":
                valeur = analyse.get("resume_couverture_pct")
            else:
                valeur = analyse.get("duree_analyse_s")
            if valeur is not None:
                valeurs.append(float(valeur))
        return np.asarray(valeurs)

    grandeurs = {
        "nb_mots": "Volume des documents déposés",
        "nb_unites": "Nombre d'unités de contenu",
        "note": "Note globale attribuée",
        "couverture": "Taux de couverture mesuré",
        "duree": "Durée de traitement",
    }
    return {
        "applique": True,
        "nb_analyses": len(analyses),
        "nb_reference": len(ancien),
        "nb_courant": len(recent),
        "grandeurs": {
            cle: {"libelle": libelle, "reference": extraire(ancien, cle),
                  "courant": extraire(recent, cle)}
            for cle, libelle in grandeurs.items()
        },
    }


def mesurer_derive() -> dict:
    """Derive de chaque grandeur surveillee, entre periode ancienne et recente."""
    donnees = _distributions_analyses()
    if not donnees.get("applique"):
        return {
            "applique": False,
            "motif": (
                f"{donnees.get('nb_analyses', 0)} analyses — au moins 12 sont "
                f"nécessaires pour comparer deux périodes"
            ),
            "grandeurs": [],
        }

    resultats = []
    for cle, contenu in donnees["grandeurs"].items():
        reference, courant = contenu["reference"], contenu["courant"]
        if len(reference) < 5 or len(courant) < 5:
            continue
        valeur = psi(reference, courant)
        niveau, lecture = _qualifier(valeur)
        resultats.append({
            "cle": cle,
            "libelle": contenu["libelle"],
            "psi": None if np.isnan(valeur) else round(valeur, 4),
            "niveau": niveau,
            "lecture": lecture,
            "ks": kolmogorov_smirnov(reference, courant),
            "moyenne_reference": round(float(np.mean(reference)), 2),
            "moyenne_courante": round(float(np.mean(courant)), 2),
            "variation_pct": (
                round(100 * (float(np.mean(courant)) - float(np.mean(reference)))
                      / abs(float(np.mean(reference))), 1)
                if abs(float(np.mean(reference))) > 1e-9 else None
            ),
        })

    resultats.sort(key=lambda r: r["psi"] if r["psi"] is not None else -1, reverse=True)
    significatives = [r for r in resultats if r["niveau"] == "significative"]

    return {
        "applique": True,
        "nb_analyses": donnees["nb_analyses"],
        "nb_reference": donnees["nb_reference"],
        "nb_courant": donnees["nb_courant"],
        "methode": "Population Stability Index + test de Kolmogorov-Smirnov",
        "seuils_psi": {"stable": PSI_STABLE, "modere": PSI_MODERE},
        "grandeurs": resultats,
        "nb_derives_significatives": len(significatives),
        "verdict": (
            "Aucune dérive significative : les modèles travaillent sur une "
            "population comparable à celle de leur entraînement."
            if not significatives else
            f"{len(significatives)} grandeur(s) ont significativement dérivé — "
            f"réentraînement recommandé."
        ),
    }


# ---------------------------------------------------------------------------
# Etat des modeles et reentrainement
# ---------------------------------------------------------------------------

def inventaire() -> dict:
    """
    Vue unifiee de tous les modeles du systeme : ceux des agents (registre)
    et ceux des modules (entraines localement).
    """
    modeles_agents = model_registry.etat()
    modeles_modules = entrainement.etat_modeles()

    a_reentrainer = []
    for modele in modeles_modules:
        raisons = []
        if modele.get("amorcage"):
            raisons.append("encore en amorçage")
        if (modele.get("age_heures") or 0) > AGE_MAX_HEURES:
            raisons.append(f"entraîné il y a plus de {AGE_MAX_HEURES} h")
        if modele.get("erreur"):
            raisons.append("dernier entraînement en échec")
        if raisons:
            a_reentrainer.append({"nom": modele["nom"], "raisons": raisons})

    return {
        "modeles_agents": modeles_agents,
        "modeles_modules": modeles_modules,
        "nb_agents": len(modeles_agents),
        "nb_modules": len(modeles_modules),
        "nb_actifs": sum(1 for m in modeles_modules if m.get("actif")),
        "nb_amorcage": sum(1 for m in modeles_modules if m.get("amorcage")),
        "a_reentrainer": a_reentrainer,
        "seuil_age_heures": AGE_MAX_HEURES,
    }


def reentrainer(noms: list[str] | None = None) -> dict:
    """
    Force le reentrainement des modeles de modules.

    L'invalidation suffit : chaque modele se reconstruit a son prochain
    usage, sur le corpus disponible a ce moment-la. Cela evite de bloquer
    une requete web le temps d'un entrainement.
    """
    debut = time.time()
    if noms:
        invalides = []
        for nom in noms:
            invalides.extend(entrainement.invalider(nom))
    else:
        invalides = entrainement.invalider()

    return {
        "modeles_invalides": invalides,
        "nb": len(invalides),
        "duree_s": round(time.time() - debut, 3),
        "message": (
            f"{len(invalides)} modèle(s) invalidé(s) : ils seront réentraînés "
            f"sur le corpus courant à leur prochaine utilisation."
        ),
    }


def tableau_de_bord() -> dict:
    """Etat complet, consomme par la page de supervision du back-office."""
    derive = mesurer_derive()
    etat = inventaire()
    return {
        "derive": derive,
        "inventaire": etat,
        "alerte": bool(
            derive.get("nb_derives_significatives") or etat.get("a_reentrainer")
        ),
        "resume": {
            "modeles_total": etat["nb_agents"] + etat["nb_modules"],
            "modeles_actifs": etat["nb_actifs"],
            "en_amorcage": etat["nb_amorcage"],
            "derives_significatives": derive.get("nb_derives_significatives", 0),
            "a_reentrainer": len(etat["a_reentrainer"]),
        },
    }
