"""
Module Historique et Tableau de Bord
=====================================

Role (rapport de conception, section 3.3.1) : offrir a l'utilisateur une
vision consolidee de l'ensemble de ses analyses, passees et en cours. Ce
module permet la recherche, le filtrage et la consultation detaillee de chaque
analyse, **dans la limite du perimetre autorise par le role de l'utilisateur**.

Deux perimetres sont donc servis par ce module :

- `tableau_de_bord_utilisateur()` : restreint aux analyses de l'utilisateur
  connecte (progression personnelle, matieres travaillees, derniers rapports) ;
- `tableau_de_bord_administrateur()` : vue globale de la plateforme (comptes,
  volume d'analyses, sante technique des composants, journal d'activite).
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from app.agents import CATALOGUE
from app.config import Config
from app.services import (
    database, extraction_documents, gemini_client, model_registry,
    profils_analyses, referentiels,
)
from app.services.vector_store import FAISS_DISPONIBLE
from app.modules import module_depot_documents


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _date(analyse: dict) -> datetime | None:
    brut = analyse.get("date_creation_iso")
    if not brut:
        return None
    try:
        valeur = datetime.fromisoformat(str(brut))
        return valeur if valeur.tzinfo else valeur.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _terminees(analyses: list[dict]) -> list[dict]:
    return [a for a in analyses if a.get("statut") == "TERMINEE"]


def filtrer(analyses: list[dict], recherche: str = "", matiere: str = "",
            statut: str = "") -> list[dict]:
    """Recherche plein texte simple + filtres, conformement au cas d'utilisation
    « consulter et filtrer l'historique »."""
    resultat = analyses
    if recherche:
        terme = recherche.strip().lower()
        resultat = [
            a for a in resultat
            if terme in str(a.get("nom_fichier", "")).lower()
            or terme in str(a.get("titre_cours", "")).lower()
            or terme in str(a.get("matiere", "")).lower()
        ]
    if matiere:
        resultat = [a for a in resultat if a.get("matiere") == matiere]
    if statut:
        resultat = [a for a in resultat if a.get("statut") == statut]
    return resultat


def _serie_temporelle(analyses: list[dict], nb_jours: int = 14) -> list[dict]:
    """Nombre d'analyses par jour sur la periode recente (graphique du dashboard)."""
    aujourdhui = datetime.now(timezone.utc).date()
    compteur = Counter()
    for analyse in analyses:
        date = _date(analyse)
        if date:
            compteur[date.date()] += 1
    return [
        {
            "date": (aujourdhui - timedelta(days=decalage)).strftime("%d/%m"),
            "valeur": compteur.get(aujourdhui - timedelta(days=decalage), 0),
        }
        for decalage in range(nb_jours - 1, -1, -1)
    ]


# ---------------------------------------------------------------------------
# Tableau de bord utilisateur
# ---------------------------------------------------------------------------

def tableau_de_bord_utilisateur(utilisateur: dict) -> dict:
    analyses = database.lister_analyses(utilisateur_id=utilisateur["id"])
    terminees = _terminees(analyses)

    notes = [float(a.get("resume_note_globale") or 0) for a in terminees]
    couvertures = [float(a.get("resume_couverture_pct") or 0) for a in terminees]

    # Progression : comparaison des 3 dernieres analyses aux 3 precedentes.
    recentes = notes[:3]
    precedentes = notes[3:6]
    if recentes and precedentes:
        evolution = round(
            sum(recentes) / len(recentes) - sum(precedentes) / len(precedentes), 1
        )
    else:
        evolution = None

    par_matiere = defaultdict(list)
    for analyse in terminees:
        par_matiere[analyse.get("matiere", "Autre")].append(
            float(analyse.get("resume_note_globale") or 0)
        )

    total_recommandations = sum(int(a.get("resume_nb_recommandations") or 0) for a in terminees)
    total_manquantes = sum(int(a.get("resume_nb_manquantes") or 0) for a in terminees)

    return {
        "nb_analyses": len(analyses),
        "nb_terminees": len(terminees),
        "nb_en_cours": sum(1 for a in analyses if a.get("statut") == "EN_COURS"),
        "nb_echecs": sum(1 for a in analyses if a.get("statut") == "ECHEC"),
        "note_moyenne": round(sum(notes) / len(notes), 1) if notes else 0.0,
        "meilleure_note": round(max(notes), 1) if notes else 0.0,
        "couverture_moyenne": round(sum(couvertures) / len(couvertures), 1) if couvertures else 0.0,
        "evolution_note": evolution,
        "total_recommandations": total_recommandations,
        "total_notions_manquantes": total_manquantes,
        "par_matiere": [
            {
                "matiere": matiere,
                "nb": len(valeurs),
                "note_moyenne": round(sum(valeurs) / len(valeurs), 1),
            }
            for matiere, valeurs in sorted(par_matiere.items(), key=lambda kv: -len(kv[1]))
        ],
        "serie_temporelle": _serie_temporelle(analyses),
        "dernieres_analyses": analyses[:6],
        "analyses": analyses,
        "matieres_utilisees": sorted({a.get("matiere", "") for a in analyses if a.get("matiere")}),
        # --- Analyse de l'historique par apprentissage --------------------
        # Les agregats ci-dessus decrivent ; ceux-ci interpretent.
        "profils": profils_analyses.regrouper(analyses),
        "trajectoire": profils_analyses.trajectoire(analyses),
        "aberrantes": profils_analyses.analyses_aberrantes(analyses),
    }


# ---------------------------------------------------------------------------
# Tableau de bord administrateur
# ---------------------------------------------------------------------------

def _sante_composants() -> list[dict]:
    """Etat des composants techniques de la plateforme (supervision)."""
    statut_db = database.statut_connexion()
    stats_llm = gemini_client.stats()

    from app.agents import agent4_vectorisation

    modele_charge = agent4_vectorisation.charger_modele_transformer() is not None

    return [
        {
            "composant": "Base de données MongoDB",
            "couche": "Couche base de données",
            "etat": "operationnel" if statut_db["disponible"] else "degrade",
            "detail": (
                f"Connectée — base « {statut_db['base']} »"
                if statut_db["disponible"]
                else f"Repli mémoire actif — {statut_db.get('erreur') or 'serveur injoignable'}"
            ),
        },
        {
            "composant": "API Gemini",
            "couche": "Couche intelligence",
            "etat": "operationnel" if stats_llm["configure"] else "hors_service",
            "detail": (
                f"Modèle {stats_llm['modele']} — {stats_llm['appels_reussis']}/"
                f"{stats_llm['appels_total']} appels réussis "
                f"({stats_llm['latence_moyenne_ms']} ms en moyenne)"
                if stats_llm["configure"]
                else "Clé API non configurée — les agents génératifs utilisent leur repli"
            ),
        },
        {
            "composant": "Modèle de vectorisation",
            "couche": "Couche intelligence",
            "etat": "operationnel" if modele_charge else "degrade",
            "detail": (
                f"sentence-transformers « {Config.EMBEDDING_MODEL} » chargé"
                if modele_charge
                else "Modèle neuronal indisponible — repli LSA (TF-IDF + SVD)"
            ),
        },
        {
            "composant": "Base vectorielle FAISS",
            "couche": "Couche intelligence",
            "etat": "operationnel" if FAISS_DISPONIBLE else "degrade",
            "detail": (
                "FAISS installé — IndexFlatIP"
                if FAISS_DISPONIBLE
                else "FAISS absent — repli NumPy exact (résultats identiques)"
            ),
        },
        {
            "composant": "Authentification Google OAuth",
            "couche": "Couche logique métier",
            "etat": "operationnel" if Config.google_oauth_configured() else "degrade",
            "detail": (
                "Client OAuth configuré"
                if Config.google_oauth_configured()
                else "Identifiants OAuth absents — connexion de démonstration active"
            ),
        },
    ]


def sante_composants() -> list[dict]:
    """Etat des composants techniques, expose aux vues (page architecture)."""
    return _sante_composants()


def tableau_de_bord_administrateur() -> dict:
    analyses = database.lister_analyses()
    utilisateurs = database.lister_utilisateurs()
    terminees = _terminees(analyses)
    notes = [float(a.get("resume_note_globale") or 0) for a in terminees]

    durees = [float(a.get("duree_analyse_s") or 0) for a in terminees if a.get("duree_analyse_s")]

    par_matiere = Counter(a.get("matiere", "Autre") for a in analyses)
    par_utilisateur = Counter(
        a.get("utilisateur_email") or "session anonyme" for a in analyses
    )

    # Taux de repli par agent : indicateur de qualite de service du pipeline.
    replis = Counter()
    for analyse in terminees:
        if (analyse.get("agent2") or {}).get("source") != "gemini":
            replis["Agent 2 — Compréhension"] += 1
        if (analyse.get("agent4") or {}).get("repli_actif"):
            replis["Agent 4 — Vectorisation"] += 1
        if not ((analyse.get("agent6") or {}).get("gemini") or {}).get("disponible"):
            replis["Agent 6 — Comparaison"] += 1
        if (analyse.get("agent8") or {}).get("source_redaction") == "repli_deterministe":
            replis["Agent 8 — Recommandations"] += 1
        if not (analyse.get("agent9") or {}).get("synthese_generee_par_ia"):
            replis["Agent 9 — Rapport final"] += 1

    return {
        "nb_utilisateurs": len(utilisateurs),
        "nb_administrateurs": sum(1 for u in utilisateurs if u.get("role") == "administrateur"),
        "nb_comptes_actifs": sum(1 for u in utilisateurs if u.get("actif", True)),
        "nb_analyses": len(analyses),
        "nb_terminees": len(terminees),
        "nb_en_cours": sum(1 for a in analyses if a.get("statut") == "EN_COURS"),
        "nb_echecs": sum(1 for a in analyses if a.get("statut") == "ECHEC"),
        "taux_reussite_pct": round(100 * len(terminees) / len(analyses), 1) if analyses else 0.0,
        "note_moyenne": round(sum(notes) / len(notes), 1) if notes else 0.0,
        "duree_moyenne_s": round(sum(durees) / len(durees), 1) if durees else 0.0,
        "par_matiere": [{"matiere": m, "nb": n} for m, n in par_matiere.most_common()],
        "top_utilisateurs": [{"email": e, "nb": n} for e, n in par_utilisateur.most_common(6)],
        "serie_temporelle": _serie_temporelle(analyses, nb_jours=21),
        "utilisateurs": utilisateurs,
        "dernieres_analyses": analyses[:12],
        "analyses": analyses,
        "replis_par_agent": [{"agent": a, "nb": n} for a, n in replis.most_common()],
        "sante": _sante_composants(),
        "llm": gemini_client.stats(),
        "stockage": module_depot_documents.statistiques_stockage(),
        "base_connaissances": referentiels.statistiques(),
        "catalogue_agents": CATALOGUE,
        "evenements": database.lister_evenements(25),
        "base_donnees": database.statut_connexion(),
        "registre_modeles": model_registry.etat(),
        "resume_modeles": model_registry.resume(),
        # Boucle de retour enseignant (plan de transition, phase 1.2) : en
        # mode ombre, ce volume est le seul effet visible du dispositif tant
        # que le seuil de bascule (~300 etiquettes de couverture) n'est pas
        # atteint.
        "retours": database.compter_retours(),
        # Lecteurs de documents : leur absence ne casse rien, mais restreint
        # silencieusement les formats acceptes. Mieux vaut l'afficher.
        "lecteurs": extraction_documents.dependances(),
    }


# ---------------------------------------------------------------------------
# Vue programme : du cours au cursus (phase 2.1 du plan de transition)
# ---------------------------------------------------------------------------
#
# Un etablissement n'evalue pas un cours, il evalue un cursus. Ce bloc assemble
# la vue d'un programme : quelles analyses sont agregeables, ce que donne leur
# agregation, et surtout **quelles analyses ont ete ecartees et pourquoi**.
#
# Le tri est aussi important que l'agregation. Melanger deux versions de
# referentiel, ou deux matieres, produirait une couverture flatteuse et fausse.
# La regle de comparabilite est la meme qu'en phase 1.1 : meme matiere, meme
# niveau, meme signature de referentiel que l'analyse la plus recente.

MOTIFS_EXCLUSION = {
    "non_terminee": "analyse non terminée",
    "matiere": "matière ou niveau différent du programme",
    "version": "version de référentiel différente",
    "introuvable": "analyse supprimée depuis son rattachement",
}


def _cle_comparabilite(analyse: dict) -> tuple:
    return (analyse.get("matiere"), analyse.get("niveau"))


def _trier_analyses(programme: dict, analyses: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Separe les analyses agregeables de celles qui ne le sont pas.

    **La matiere et le niveau viennent du programme lui-meme**, pas des
    analyses. Les deduire de l'analyse la plus recente rendrait le perimetre
    du cursus dependant de l'ordre des depots : rattacher un document de
    sciences a un programme de mathematiques ecarterait les mathematiques.
    Le programme declare ce qu'il est ; les documents s'y conforment ou non.

    La version de referentiel, elle, n'est pas declarable a l'avance : elle
    est celle de l'analyse retenue la plus recente, comme pour la trajectoire.
    """
    terminees = [a for a in analyses if a.get("statut") == "TERMINEE"]
    ecartees = [
        {**a, "motif_exclusion": MOTIFS_EXCLUSION["non_terminee"]}
        for a in analyses if a.get("statut") != "TERMINEE"
    ]
    if not terminees:
        return [], ecartees

    terminees.sort(key=lambda a: str(a.get("date_creation_iso") or ""))

    cle_programme = (programme.get("matiere"), programme.get("niveau"))
    if not all(cle_programme):
        # Programme cree sans matiere ni niveau : on se rabat sur l'analyse la
        # plus recente, faute de mieux.
        cle_programme = _cle_comparabilite(terminees[-1])

    conformes = []
    for analyse in terminees:
        if _cle_comparabilite(analyse) == cle_programme:
            conformes.append(analyse)
        else:
            ecartees.append({**analyse, "motif_exclusion": MOTIFS_EXCLUSION["matiere"]})

    if not conformes:
        return [], ecartees

    version_reference = conformes[-1].get("referentiel_version")
    retenues = []
    for analyse in conformes:
        if analyse.get("referentiel_version") != version_reference:
            ecartees.append({**analyse, "motif_exclusion": MOTIFS_EXCLUSION["version"]})
        else:
            retenues.append(analyse)
    return retenues, ecartees


def _matrice_notions_documents(agregat: dict, documents: list[dict]) -> dict:
    """
    Matrice notions x documents : qui traite quoi, d'un coup d'oeil.

    C'est la vue qui rend la repartition entre enseignants discutable — une
    notion dont toutes les cellules sont faibles est un trou reel du cursus,
    la ou une notion couverte par un seul document signale une dependance.
    """
    ordre = [d["analyse_id"] for d in documents]
    lignes = []
    for notion in agregat.get("notions", []):
        par_analyse = {c["analyse_id"]: c for c in notion.get("contributeurs", [])}
        lignes.append({
            "notion": notion["notion"],
            "cle_notion": notion["cle_notion"],
            "pays": notion["pays"],
            "code": notion["code"],
            "statut": notion["statut"],
            "libelle_ecart": notion["libelle_ecart"],
            "probabilite": notion["probabilite_couverture"],
            "cellules": [
                {
                    "analyse_id": analyse_id,
                    "probabilite": (par_analyse.get(analyse_id) or {}).get("probabilite", 0.0),
                    "statut": (par_analyse.get(analyse_id) or {}).get("statut", "Non couverte"),
                }
                for analyse_id in ordre
            ],
        })
    return {
        "documents": documents,
        "lignes": lignes,
        "nb_lignes": len(lignes),
    }


def vue_programme(programme: dict) -> dict:
    """
    Contexte d'affichage d'un programme : couverture agregee, matrice notions
    x documents, apport de chaque document, et analyses ecartees avec motif.
    """
    from app.agents import agent6_comparaison

    analyses = database.analyses_du_programme(programme)
    identifiants_connus = {a.get("id") for a in analyses}
    orphelines = [
        {"id": identifiant, "nom_fichier": "—",
         "motif_exclusion": MOTIFS_EXCLUSION["introuvable"]}
        for identifiant in (programme.get("analyse_ids") or [])
        if identifiant not in identifiants_connus
    ]

    retenues, ecartees = _trier_analyses(programme, analyses)
    ecartees += orphelines

    agregat = agent6_comparaison.agreger_programme(retenues)

    documents = agregat.get("couverture_par_document", [])
    matrice = _matrice_notions_documents(agregat, documents) if agregat.get("disponible") else {
        "documents": [], "lignes": [], "nb_lignes": 0
    }

    # Gains de l'agregation : le nombre d'ecarts qui disparaissent une fois le
    # cursus considere dans son ensemble. C'est l'argument central de cette
    # vue — beaucoup d'ecarts d'un document isole n'en sont pas.
    manquantes_isolees = sum(
        (a.get("agent6") or {}).get("nb_notions_manquantes", 0) for a in retenues
    )
    manquantes_agregees = agregat.get("nb_notions_manquantes", 0)

    reference = retenues[-1] if retenues else None
    return {
        "programme": programme,
        "analyses": retenues,
        "nb_analyses": len(retenues),
        "analyses_ecartees": ecartees,
        "nb_ecartees": len(ecartees),
        "agregat": agregat,
        "matrice": matrice,
        "documents": documents,
        "referentiel_version": (reference or {}).get("referentiel_version"),
        "referentiel_officiel": bool((reference or {}).get("referentiel_officiel")),
        "matiere": (reference or {}).get("matiere") or programme.get("matiere"),
        "niveau": (reference or {}).get("niveau") or programme.get("niveau"),
        "ecarts_resorbes": max(0, manquantes_isolees - manquantes_agregees),
        "manquantes_cumulees_isolement": manquantes_isolees,
        "note_moyenne": (
            round(sum(float(a.get("resume_note_globale") or 0) for a in retenues)
                  / len(retenues), 1)
            if retenues else 0.0
        ),
        "enseignants": sorted({
            d["enseignant"] for d in documents if d.get("enseignant")
        }),
    }


def analyses_rattachables(programme: dict, utilisateur: dict | None) -> list[dict]:
    """
    Analyses que l'utilisateur peut ajouter a ce programme : les siennes (ou
    toutes, s'il est administrateur), terminees, pas deja rattachees.
    """
    deja = set(programme.get("analyse_ids") or [])
    if utilisateur and utilisateur.get("role") == "administrateur":
        candidates = database.lister_analyses()
    else:
        candidates = database.lister_analyses(
            utilisateur_id=(utilisateur or {}).get("id")
        )
    return [
        a for a in candidates
        if a.get("statut") == "TERMINEE" and a.get("id") not in deja
    ]
