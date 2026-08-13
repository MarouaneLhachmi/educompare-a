"""
Agent 3 — Decoupage
====================

Role (rapport de conception, section 3.3.2) : decouper le contenu du cours en
unites de sens coherentes, respectant les frontieres naturelles des idees
exprimees, afin de preparer les etapes de comparaison semantique.

Nature de l'agent : **deterministe (algorithmique)**. Le decoupage suit une
strategie de segmentation hierarchique classique en RAG :

1. la frontiere de plus haut niveau est le chapitre detecte par l'Agent 1 —
   on ne fusionne jamais deux chapitres dans une meme unite ;
2. a l'interieur d'un chapitre, le texte est segmente en phrases, puis les
   phrases sont agregees jusqu'a atteindre une taille cible en mots ;
3. un **recouvrement** (overlap) de quelques mots est conserve entre deux
   unites consecutives, afin qu'une idee a cheval sur une frontiere reste
   representee dans au moins un segment complet ;
4. les segments trop courts sont fusionnes avec leur voisin pour eviter les
   vecteurs peu informatifs qui degraderaient la recherche de l'Agent 5.

Entree : sorties des Agents 1 et 2
Sortie : liste d'unites de sens pretes a etre vectorisees
"""

import re

TAILLE_CIBLE_MOTS = 110
TAILLE_MIN_MOTS = 25
TAILLE_MAX_MOTS = 180
RECOUVREMENT_MOTS = 20

_FIN_DE_PHRASE = re.compile(r"(?<=[.!?:;])\s+(?=[A-ZÉÈÀÂÎÔÛÇ0-9])")


def _segmenter_en_phrases(texte: str) -> list[str]:
    """Segmentation en phrases par ponctuation forte, avec repli sur les sauts de ligne."""
    texte = re.sub(r"\s+", " ", texte or "").strip()
    if not texte:
        return []
    phrases = [p.strip() for p in _FIN_DE_PHRASE.split(texte) if p.strip()]
    if len(phrases) <= 1 and len(texte.split()) > TAILLE_MAX_MOTS:
        # Aucun signe de ponctuation exploitable (PDF mal structure) :
        # on retombe sur un decoupage par blocs de mots.
        mots = texte.split()
        phrases = [
            " ".join(mots[i : i + TAILLE_CIBLE_MOTS])
            for i in range(0, len(mots), TAILLE_CIBLE_MOTS)
        ]
    return phrases


def _agreger(phrases: list[str]) -> list[str]:
    """Agrege les phrases en unites proches de la taille cible, avec recouvrement."""
    unites: list[str] = []
    courant: list[str] = []
    nb_mots = 0

    for phrase in phrases:
        mots_phrase = len(phrase.split())
        if nb_mots + mots_phrase > TAILLE_MAX_MOTS and courant:
            unites.append(" ".join(courant))
            queue = " ".join(courant).split()[-RECOUVREMENT_MOTS:]
            courant = [" ".join(queue)] if queue else []
            nb_mots = len(queue)
        courant.append(phrase)
        nb_mots += mots_phrase
        if nb_mots >= TAILLE_CIBLE_MOTS:
            unites.append(" ".join(courant))
            queue = " ".join(courant).split()[-RECOUVREMENT_MOTS:]
            courant = [" ".join(queue)] if queue else []
            nb_mots = len(queue)

    reste = " ".join(courant).strip()
    if reste:
        if unites and len(reste.split()) < TAILLE_MIN_MOTS:
            unites[-1] = unites[-1] + " " + reste
        else:
            unites.append(reste)
    return unites


def _objectifs_du_chapitre(agent2: dict, titre: str) -> list[str]:
    for chapitre in agent2.get("chapitres", []):
        if chapitre.get("titre", "").strip().lower() == titre.strip().lower():
            return chapitre.get("objectifs_pedagogiques", [])
    return []


def process(agent1: dict, agent2: dict) -> dict:
    """
    Execute l'Agent 3.

    Retourne :
    {
        "strategie": str,
        "parametres": {...},
        "unites": [
            {"id", "chapitre", "page", "position", "texte", "nb_mots", "objectifs"}
        ],
        "nb_unites": int,
        "taille_moyenne_mots": float
    }
    """
    unites = []
    compteur = 0

    for chapitre in agent1.get("chapitres", []):
        titre = chapitre.get("titre", "Section")
        contenu = chapitre.get("contenu") or chapitre.get("extrait") or ""
        # Le titre est prefixe au premier segment : il porte une forte charge
        # semantique et ameliore sensiblement la recherche de l'Agent 5.
        segments = _agreger(_segmenter_en_phrases(contenu)) or [titre]
        objectifs = _objectifs_du_chapitre(agent2, titre)

        for position, segment in enumerate(segments):
            texte = f"{titre}. {segment}" if position == 0 else segment
            compteur += 1
            unites.append(
                {
                    "id": f"u{compteur:03d}",
                    "chapitre": titre,
                    "page": chapitre.get("page"),
                    "position": position,
                    "texte": texte.strip(),
                    "nb_mots": len(texte.split()),
                    "objectifs": objectifs,
                }
            )

    if not unites:
        # Document sans structure exploitable : une unite unique construite a
        # partir des notions cles identifiees par l'Agent 2.
        secours = ". ".join(agent2.get("notions_cles_globales", [])) or agent1.get(
            "texte_brut_tronque", ""
        )
        unites = [
            {
                "id": "u001",
                "chapitre": agent2.get("titre_cours", "Document"),
                "page": 1,
                "position": 0,
                "texte": secours[:1200],
                "nb_mots": len(secours.split()),
                "objectifs": [],
            }
        ]

    tailles = [u["nb_mots"] for u in unites]
    return {
        "strategie": "segmentation hierarchique chapitre -> phrases -> agregation avec recouvrement",
        "parametres": {
            "taille_cible_mots": TAILLE_CIBLE_MOTS,
            "taille_min_mots": TAILLE_MIN_MOTS,
            "taille_max_mots": TAILLE_MAX_MOTS,
            "recouvrement_mots": RECOUVREMENT_MOTS,
        },
        "unites": unites,
        "nb_unites": len(unites),
        "taille_moyenne_mots": round(sum(tailles) / len(tailles), 1) if tailles else 0.0,
    }
