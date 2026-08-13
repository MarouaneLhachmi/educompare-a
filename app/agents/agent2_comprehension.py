"""
Agent 2 — Comprehension
========================

Role (rapport de conception, section 3.3.2) : analyser le contenu textuel
extrait par l'Agent 1 afin d'en degager la structure pedagogique sous-jacente :
titre, chapitres, objectifs pedagogiques, notions cles et niveau academique
estime.

Nature de l'agent : **hybride**. Il combine :
- un appel a un modele de langage (API Gemini) pour la lecture semantique du
  document et la formulation des objectifs pedagogiques implicites ;
- un repli deterministe integral (heuristiques lexicales sur la sortie de
  l'Agent 1) garantissant que le pipeline aboutit meme sans acces reseau.

Entree : sortie de l'Agent 1 + matiere et niveau declares
Sortie : structure pedagogique normalisee (voir `process()`)
"""

import re

from app.services import gemini_client

PROMPT = """Tu es un ingenieur pedagogique charge d'analyser un support de cours.

Matiere declaree : {matiere}
Niveau declare   : {niveau}
Langue detectee  : {langue}

Titres de sections detectes automatiquement dans le document :
{titres}

Mots les plus frequents du document :
{mots_cles}

Extrait du debut du document (brut) :
\"\"\"{extrait}\"\"\"

Analyse ce support et produis sa structure pedagogique.

Reponds UNIQUEMENT avec un objet JSON valide, sans texte ni balise autour,
avec exactement cette structure :
{{
  "titre_cours": "titre synthetique du cours (une ligne)",
  "resume": "resume du contenu du cours en 2 a 3 phrases",
  "discipline_identifiee": "matiere reellement traitee dans le document",
  "niveau_estime": "niveau academique estime a la lecture du contenu",
  "coherence_niveau_declare": "coherent" | "legerement_decale" | "incoherent",
  "chapitres": [
    {{
      "titre": "titre du chapitre",
      "objectifs_pedagogiques": ["objectif 1", "objectif 2"],
      "notions_cles": ["notion 1", "notion 2"]
    }}
  ],
  "notions_cles_globales": ["notion", "notion", "..."],
  "prerequis": ["prerequis 1", "prerequis 2"],
  "competences_visees": ["competence 1", "competence 2"]
}}

Contraintes : entre 3 et 12 chapitres, entre 8 et 20 notions cles globales,
formulations courtes, en francais.
"""

# Verbes d'action frequemment utilises pour formuler un objectif pedagogique.
VERBES_OBJECTIFS = [
    "calculer", "identifier", "reconnaitre", "reconnaître", "resoudre", "résoudre",
    "comparer", "construire", "mesurer", "utiliser", "decrire", "décrire",
    "expliquer", "classer", "tracer", "convertir", "analyser", "appliquer",
]


def _fabriquer_repli(agent1: dict, matiere: str, niveau: str, motif: str) -> dict:
    """
    Repli deterministe : reconstruit une structure pedagogique plausible a
    partir des seules donnees de l'Agent 1, sans aucun appel reseau.
    """
    chapitres = []
    for chapitre in agent1.get("chapitres", []):
        contenu = chapitre.get("contenu", "")
        phrases = [p.strip() for p in re.split(r"[.\n;]", contenu) if 25 < len(p.strip()) < 180]
        objectifs = [
            p for p in phrases
            if any(verbe in p.lower() for verbe in VERBES_OBJECTIFS)
        ][:3]
        if not objectifs:
            objectifs = [f"Maitriser les notions abordees dans « {chapitre['titre']} »."]
        mots_chapitre = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{5,}", contenu.lower())
        notions = []
        for mot in mots_chapitre:
            if mot not in notions:
                notions.append(mot)
            if len(notions) >= 5:
                break
        chapitres.append(
            {
                "titre": chapitre["titre"],
                "objectifs_pedagogiques": objectifs,
                "notions_cles": notions,
            }
        )

    notions_globales = [m["mot"] for m in agent1.get("mots_cles", [])[:15]]
    titre = (agent1.get("metadonnees", {}) or {}).get("titre_pdf") or (
        f"Cours de {matiere} — {niveau}"
    )

    return {
        "titre_cours": titre,
        "resume": (
            f"Support de {matiere} comportant {len(chapitres)} section(s) et "
            f"{agent1.get('nb_mots', 0)} mots, structure autour des notions suivantes : "
            + ", ".join(notions_globales[:6])
            + "."
        ),
        "discipline_identifiee": matiere,
        "niveau_estime": niveau,
        "coherence_niveau_declare": "coherent",
        "chapitres": chapitres,
        "notions_cles_globales": notions_globales,
        "prerequis": [],
        "competences_visees": [],
        "source": "repli_deterministe",
        "motif_repli": motif,
    }


def _normaliser(donnees: dict, agent1: dict, matiere: str, niveau: str) -> dict:
    """Securise la sortie du modele : types attendus, valeurs par defaut."""
    def liste(valeur):
        if isinstance(valeur, list):
            return [str(v).strip() for v in valeur if str(v).strip()]
        if isinstance(valeur, str) and valeur.strip():
            return [valeur.strip()]
        return []

    chapitres = []
    for chapitre in donnees.get("chapitres") or []:
        if not isinstance(chapitre, dict):
            continue
        chapitres.append(
            {
                "titre": str(chapitre.get("titre", "Chapitre sans titre")).strip(),
                "objectifs_pedagogiques": liste(chapitre.get("objectifs_pedagogiques")),
                "notions_cles": liste(chapitre.get("notions_cles")),
            }
        )
    if not chapitres:
        chapitres = [
            {"titre": c["titre"], "objectifs_pedagogiques": [], "notions_cles": []}
            for c in agent1.get("chapitres", [])
        ]

    coherence = str(donnees.get("coherence_niveau_declare", "coherent")).strip().lower()
    if coherence not in {"coherent", "legerement_decale", "incoherent"}:
        coherence = "coherent"

    return {
        "titre_cours": str(donnees.get("titre_cours") or f"Cours de {matiere}").strip(),
        "resume": str(donnees.get("resume") or "").strip(),
        "discipline_identifiee": str(donnees.get("discipline_identifiee") or matiere).strip(),
        "niveau_estime": str(donnees.get("niveau_estime") or niveau).strip(),
        "coherence_niveau_declare": coherence,
        "chapitres": chapitres,
        "notions_cles_globales": liste(donnees.get("notions_cles_globales")),
        "prerequis": liste(donnees.get("prerequis")),
        "competences_visees": liste(donnees.get("competences_visees")),
        "source": "gemini",
        "motif_repli": None,
    }


def process(agent1: dict, matiere: str, niveau: str) -> dict:
    """Execute l'Agent 2 : LLM si disponible, repli deterministe sinon."""
    titres = "\n".join(f"- {c['titre']}" for c in agent1.get("chapitres", [])) or "(aucun)"
    mots = ", ".join(m["mot"] for m in agent1.get("mots_cles", [])[:20]) or "(aucun)"

    prompt = PROMPT.format(
        matiere=matiere,
        niveau=niveau,
        langue=agent1.get("langue_detectee", "fr"),
        titres=titres[:2500],
        mots_cles=mots,
        extrait=agent1.get("texte_brut_tronque", "")[:2500],
    )

    try:
        donnees = gemini_client.generate_json(prompt, agent="agent2_comprehension")
        if not isinstance(donnees, dict):
            raise ValueError("structure JSON inattendue")
        return _normaliser(donnees, agent1, matiere, niveau)
    except Exception as exc:
        return _fabriquer_repli(agent1, matiere, niveau, str(exc))
