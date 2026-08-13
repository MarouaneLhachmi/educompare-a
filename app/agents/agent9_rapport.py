"""
Agent 9 — Rapport final
========================

Role (rapport de conception, section 3.3.2) : agreger l'ensemble des resultats
produits par les agents precedents en un rapport pedagogique final, coherent
et directement exploitable, incluant une **synthese executive**.

Nature de l'agent : **hybride**.

- L'**agregation est deterministe** : l'agent construit les chiffres cles, le
  classement des referentiels, la table des matieres du rapport et les
  indicateurs de fiabilite de l'analyse (quels agents ont reellement abouti,
  quels replis ont ete actives). Aucune valeur affichee dans le rapport n'est
  inventee par un modele : toutes proviennent des agents amont.
- La **synthese executive est generative** : le modele de langage redige la
  page de garde destinee a un decideur (responsable pedagogique, comite
  d'accreditation). Elle est explicitement identifiee comme telle dans
  l'interface, et un repli deterministe la remplace si l'API est indisponible.

Entree : sorties des Agents 1 a 8
Sortie : rapport final structure (voir `process()`)
"""

from app.services import gemini_client

PROMPT = """Tu rediges la synthese executive d'un rapport d'analyse pedagogique
destine a un responsable d'etablissement engage dans une demarche d'accreditation.

Cours analyse : {titre_cours}
Matiere : {matiere} — Niveau : {niveau}
Resume du contenu : {resume}

Resultats de l'analyse automatique :
- Note globale d'alignement international : {note}/100 (niveau « {maturite} »)
- Couverture par referentiel : {couverture}
- Notions attendues non couvertes : {nb_manquantes}
- Contenus specifiques au cours sans equivalent etranger : {nb_excedentaires}
- Recommandations prioritaires : {recos}

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour :
{{
  "titre_rapport": "titre du rapport (une ligne)",
  "synthese_executive": "5 a 7 phrases a destination d'un decideur non technique",
  "verdict": "une phrase tranchee sur le positionnement du cours",
  "chiffres_cles_commentes": [
    {{"libelle": "nom du chiffre", "valeur": "valeur", "lecture": "ce qu'il faut en retenir"}}
  ],
  "prochaines_etapes": ["etape 1", "etape 2", "etape 3"],
  "message_enseignant": "2 phrases bienveillantes et constructives adressees a l'enseignant"
}}

Contraintes : francais, ton professionnel, 3 a 5 chiffres cles, pas de jargon technique.
"""


def _classement_referentiels(agent6: dict) -> list[dict]:
    pays = list((agent6.get("par_pays") or {}).values())
    pays.sort(key=lambda p: p["taux_couverture_pct"], reverse=True)
    return [
        {
            "rang": i + 1,
            "code": p["code"],
            "pays": p["pays"],
            "drapeau": p.get("drapeau", ""),
            "referentiel": p.get("referentiel", ""),
            "taux_couverture_pct": p["taux_couverture_pct"],
            "nb_couvertes": p["nb_couvertes"],
            "nb_partielles": p["nb_partielles"],
            "nb_manquantes": p["nb_manquantes"],
            "nb_notions": p["nb_notions"],
        }
        for i, p in enumerate(pays)
    ]


def _fiabilite(agent1: dict, agent2: dict, agent4: dict, agent6: dict,
               agent7: dict, agent8: dict) -> dict:
    """
    Indicateurs de fiabilite : l'utilisateur doit savoir dans quelles
    conditions l'analyse a ete produite (replis actives, API indisponible...).
    """
    alertes = []
    if agent1.get("ocr_utilise"):
        alertes.append(
            "Le texte a ete reconstitue par reconnaissance optique : des erreurs "
            "de lecture peuvent affecter la comparaison."
        )
    if agent4.get("repli_actif"):
        alertes.append(
            "Le modele neuronal de vectorisation n'etait pas disponible : le repli "
            "statistique (LSA) est moins performant sur les referentiels rediges en anglais."
        )
    if agent2.get("source") != "gemini":
        alertes.append("La structure pedagogique a ete deduite par regles, sans modele de langage.")
    if not (agent6.get("gemini") or {}).get("disponible"):
        alertes.append("L'analyse qualitative comparative n'a pas pu etre generee.")
    if agent8.get("source_redaction") == "repli_deterministe":
        alertes.append("Les recommandations ont ete produites par gabarit, sans redaction assistee.")

    niveau_confiance = max(0, 100 - 15 * len(alertes))
    return {
        "niveau_confiance_pct": niveau_confiance,
        "alertes": alertes,
        "moteur_vectorisation": agent4.get("moteur"),
        "type_moteur": agent4.get("type_moteur"),
        "modele_evaluation": (agent7.get("modele") or {}).get("algorithme"),
    }


def _repli_synthese(agent2: dict, agent6: dict, agent7: dict, agent8: dict,
                    matiere: str, niveau: str) -> dict:
    classement = _classement_referentiels(agent6)
    meilleur = classement[0] if classement else None
    return {
        "titre_rapport": f"Rapport d'analyse comparative — {agent2.get('titre_cours', matiere)}",
        "synthese_executive": (
            f"Le support de {matiere} destine au niveau « {niveau} » obtient une note globale "
            f"d'alignement international de {agent7.get('note_globale', 0)}/100, correspondant au "
            f"niveau « {agent7.get('niveau_maturite', '')} ». "
            + (
                f"Le referentiel dont il est le plus proche est celui de {meilleur['pays']}, "
                f"avec {meilleur['taux_couverture_pct']} % de couverture. "
                if meilleur
                else ""
            )
            + f"L'analyse identifie {agent6.get('nb_notions_manquantes', 0)} notion(s) attendue(s) "
            f"non couverte(s) et propose {len(agent8.get('recommandations', []))} recommandation(s) "
            f"priorisee(s). Cette synthese a ete generee automatiquement a partir des resultats "
            f"chiffres, le modele de langage n'etant pas disponible."
        ),
        "verdict": agent7.get("message_maturite", ""),
        "chiffres_cles_commentes": [],
        "prochaines_etapes": [r["titre"] for r in agent8.get("recommandations", [])[:3]],
        "message_enseignant": (
            "Les ecarts releves sont des pistes d'evolution, pas un jugement sur la qualite "
            "de l'enseignement dispense. Les recommandations sont classees par priorite pour "
            "faciliter leur mise en oeuvre progressive."
        ),
        "disponible": False,
    }


def process(agent1: dict, agent2: dict, agent3: dict, agent4: dict, agent5: dict,
            agent6: dict, agent7: dict, agent8: dict, matiere: str, niveau: str) -> dict:
    """Execute l'Agent 9 : agregation + synthese executive."""
    classement = _classement_referentiels(agent6)

    chiffres_cles = [
        {
            "cle": "note_globale",
            "libelle": "Note d'alignement international",
            "valeur": agent7.get("note_globale", 0),
            "unite": "/100",
        },
        {
            "cle": "couverture_moyenne",
            "libelle": "Couverture moyenne des referentiels",
            "valeur": agent6.get("score_global_pct", 0),
            "unite": "%",
        },
        {
            "cle": "notions_manquantes",
            "libelle": "Notions attendues non couvertes",
            "valeur": agent6.get("nb_notions_manquantes", 0),
            "unite": "",
        },
        {
            "cle": "recommandations",
            "libelle": "Recommandations priorisees",
            "valeur": len(agent8.get("recommandations", [])),
            "unite": "",
        },
        {
            "cle": "unites_analysees",
            "libelle": "Unites de sens analysees",
            "valeur": agent3.get("nb_unites", 0),
            "unite": "",
        },
        {
            "cle": "notions_comparees",
            "libelle": "Notions de referentiels comparees",
            "valeur": agent5.get("nb_notions_indexees", 0),
            "unite": "",
        },
    ]

    recos_txt = " | ".join(r["titre"] for r in agent8.get("recommandations", [])[:5]) or "(aucune)"
    couverture_txt = ", ".join(
        f"{p['pays']} {p['taux_couverture_pct']}%" for p in classement
    ) or "(aucun referentiel)"

    prompt = PROMPT.format(
        titre_cours=agent2.get("titre_cours", ""),
        matiere=matiere,
        niveau=niveau,
        resume=agent2.get("resume", ""),
        note=agent7.get("note_globale", 0),
        maturite=agent7.get("niveau_maturite", ""),
        couverture=couverture_txt,
        nb_manquantes=agent6.get("nb_notions_manquantes", 0),
        nb_excedentaires=agent6.get("nb_contenus_excedentaires", 0),
        recos=recos_txt,
    )

    try:
        synthese = gemini_client.generate_json(prompt, agent="agent9_rapport")
        if not isinstance(synthese, dict) or not synthese.get("synthese_executive"):
            raise ValueError("structure JSON inattendue")
        synthese["disponible"] = True
    except Exception:
        synthese = _repli_synthese(agent2, agent6, agent7, agent8, matiere, niveau)

    return {
        "titre_rapport": str(
            synthese.get("titre_rapport")
            or f"Rapport d'analyse — {agent2.get('titre_cours', matiere)}"
        ),
        "synthese_executive": str(synthese.get("synthese_executive", "")),
        "verdict": str(synthese.get("verdict") or agent7.get("message_maturite", "")),
        "message_enseignant": str(synthese.get("message_enseignant", "")),
        "chiffres_cles": chiffres_cles,
        "chiffres_cles_commentes": list(synthese.get("chiffres_cles_commentes") or []),
        "prochaines_etapes": list(synthese.get("prochaines_etapes") or []),
        "classement_referentiels": classement,
        "fiabilite": _fiabilite(agent1, agent2, agent4, agent6, agent7, agent8),
        "synthese_generee_par_ia": bool(synthese.get("disponible")),
        "sommaire": [
            "Synthese executive",
            "Structure du cours (Agents 1 a 3)",
            "Recherche semantique et comparaison (Agents 4 a 6)",
            "Evaluation pedagogique (Agent 7)",
            "Recommandations priorisees (Agent 8)",
        ],
    }
