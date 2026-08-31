"""
Agent 1 — Extraction
=====================

Role (rapport de conception, section 3.3.2) : point d'entree du pipeline de
traitement. Il lit le document depose et en extrait le contenu textuel
exploitable par les etapes suivantes, y compris pour les documents ne
contenant pas de texte nativement numerique (documents scannes).

Nature de l'agent : **purement deterministe (reactif)**. Aucun modele
d'apprentissage n'intervient ici : l'extraction repose sur la couche texte du
PDF, des expressions regulieres et des statistiques lexicales. C'est un choix
assume : le premier maillon de la chaine doit etre rapide, reproductible et
independant de toute API externe.

Strategie par format (plan de transition, phase 2.2) : la lecture proprement
dite est deleguee a `services/extraction_documents`, qui sait ouvrir un PDF,
un document Word ou une presentation, et basculer sur l'OCR quand un PDF n'a
aucune couche texte. L'agent garde ce qui releve de sa competence propre :
reperer la structure, compter, caracteriser.

Structure declaree contre structure devinee : un PDF n'indique pas ou
commencent ses chapitres — il faut les deviner par expressions regulieres. Un
`.docx` porte ses styles de titre, un `.pptx` ses titres de diapositive.
Quand cette structure existe, l'agent la prefere a son heuristique : une
structure connue vaut mieux qu'une structure devinee.

Entree : chemin d'un document (.pdf, .docx, .pptx)
Sortie : dictionnaire structure (voir `process()`)
"""

import re
from collections import Counter

from app.services import extraction_documents
from app.services.extraction_documents import DocumentIllisible

# Mots vides francais et anglais (liste courte embarquee : evite une
# dependance a un corpus externe telechargeable).
STOPWORDS = {
    # francais
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "et", "ou",
    "en", "dans", "sur", "pour", "par", "avec", "sans", "ce", "cet", "cette",
    "ces", "son", "sa", "ses", "leur", "leurs", "il", "elle", "ils", "elles",
    "on", "nous", "vous", "je", "tu", "est", "sont", "etre", "être", "avoir",
    "que", "qui", "quoi", "dont", "où", "ou", "se", "ne", "pas", "plus",
    "comme", "mais", "donc", "car", "si", "tout", "tous", "toute", "toutes",
    "aussi", "entre", "vers", "chez", "sous", "notre", "votre", "afin",
    "ainsi", "chapitre", "leçon", "lecon", "unité", "unite", "exercice",
    "exercices", "page", "partie", "cours", "année", "annee", "puis", "peut",
    "faire", "fait", "deux", "trois", "elle", "cela", "celui", "meme", "même",
    # anglais
    "the", "and", "for", "with", "this", "that", "from", "they", "have", "has",
    "are", "was", "were", "will", "can", "not", "you", "your", "its", "their",
    "into", "than", "then", "them", "these", "those", "such", "each", "which",
}

# Motifs de detection de titres de chapitre / lecon (heuristique, sans IA).
CHAPTER_PATTERNS = [
    re.compile(
        r"^\s*(chapitre|leçon|lecon|unité|unite|partie|module|séquence|sequence|thème|theme)\b"
        r"\s*[:\-\.]?\s*(\d+)?\s*[:\-]?\s*(.*)$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(\d{1,2})[\.\)]\s+([A-ZÉÈÀÂÎÔÛÇ][^\n]{3,80})$"),
    re.compile(r"^\s*(chapter|lesson|unit|section)\b\s*(\d+)?\s*[:\-]?\s*(.*)$", re.IGNORECASE),
]

MIN_TITLE_LEN = 3
MAX_TITLE_LEN = 90


# ---------------------------------------------------------------------------
# Lecture du document
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Analyse structurelle (regles)
# ---------------------------------------------------------------------------

def _detecter_chapitres(pages: list[str], titres_declares: list[str] | None = None) -> list[dict]:
    """
    Detecte les titres de chapitre/lecon ligne par ligne, et associe a chaque
    titre le corps de texte qui le suit jusqu'au titre suivant, ainsi que la
    page ou il apparait.

    `titres_declares` porte la structure que l'auteur a lui-meme balisee
    (styles de titre d'un .docx, titres de diapositive d'un .pptx). Une ligne
    qui y figure est un titre, sans avoir a correspondre a un motif : ces
    documents nomment rarement leurs sections « Chapitre 3 », et l'heuristique
    seule les manquerait.
    """
    declares = {t.strip() for t in (titres_declares or []) if t and t.strip()}
    lignes: list[tuple[str, int]] = []
    for numero_page, texte in enumerate(pages, start=1):
        for ligne in texte.splitlines():
            lignes.append((ligne.strip(), numero_page))

    reperages: list[tuple[int, str, int]] = []  # (index, titre, page)
    for index, (ligne, page) in enumerate(lignes):
        if not ligne:
            continue
        if ligne in declares:
            if len(ligne) >= MIN_TITLE_LEN:
                reperages.append((index, ligne, page))
            continue
        if len(ligne) > MAX_TITLE_LEN:
            continue
        for motif in CHAPTER_PATTERNS:
            if motif.match(ligne):
                titre = ligne.strip(" :-.\t")
                if len(titre) >= MIN_TITLE_LEN:
                    reperages.append((index, titre, page))
                break

    chapitres = []
    if reperages:
        for i, (index, titre, page) in enumerate(reperages):
            fin = reperages[i + 1][0] if i + 1 < len(reperages) else len(lignes)
            corps = [l for l, _ in lignes[index + 1 : fin] if l]
            contenu = " ".join(corps)
            chapitres.append(
                {
                    "titre": titre,
                    "extrait": contenu[:600],
                    "contenu": contenu,
                    "page": page,
                    "nb_mots": len(contenu.split()),
                }
            )
    else:
        # Repli : aucun titre detecte -> decoupage mecanique en sections egales
        non_vides = [(l, p) for l, p in lignes if l]
        nb_sections = min(6, max(1, len(non_vides) // 25 or 1))
        if non_vides:
            taille = max(1, len(non_vides) // nb_sections)
            for i in range(nb_sections):
                bloc = non_vides[i * taille : (i + 1) * taille]
                if not bloc:
                    continue
                contenu = " ".join(l for l, _ in bloc)
                chapitres.append(
                    {
                        "titre": f"Section {i + 1} (titre non detecte automatiquement)",
                        "extrait": contenu[:600],
                        "contenu": contenu,
                        "page": bloc[0][1],
                        "nb_mots": len(contenu.split()),
                    }
                )
    return chapitres


def _extraire_mots_cles(texte: str, top_n: int = 25) -> list[dict]:
    """Frequence lexicale simple (sans IA), hors mots vides et nombres seuls."""
    mots = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", texte.lower())
    filtres = [m for m in mots if m not in STOPWORDS]
    compteur = Counter(filtres)
    total = sum(compteur.values()) or 1
    return [
        {"mot": mot, "occurrences": n, "frequence_pct": round(100 * n / total, 2)}
        for mot, n in compteur.most_common(top_n)
    ]


def _detecter_langue(texte: str) -> str:
    """Heuristique FR/EN basee sur la presence de marqueurs frequents."""
    minuscule = texte.lower()
    marqueurs_fr = [" le ", " la ", " les ", " et ", " un ", " une ", " des ", " dans ", " est "]
    marqueurs_en = [" the ", " and ", " is ", " are ", " of ", " to ", " with "]
    score_fr = sum(minuscule.count(m) for m in marqueurs_fr)
    score_en = sum(minuscule.count(m) for m in marqueurs_en)
    return "fr" if score_fr >= score_en else "en"


def _compter_elements_pedagogiques(texte: str) -> dict:
    """Reperage lexical d'elements structurants du support de cours."""
    minuscule = texte.lower()
    return {
        "exercices": len(re.findall(r"\bexercices?\b", minuscule)),
        "exemples": len(re.findall(r"\bexemples?\b", minuscule)),
        "definitions": len(re.findall(r"\bd[ée]finitions?\b", minuscule)),
        "objectifs": len(re.findall(r"\bobjectifs?\b", minuscule)),
        "evaluations": len(re.findall(r"\b[ée]valuations?|contr[ôo]les?\b", minuscule)),
        "figures": len(re.findall(r"\bfigures?|sch[ée]mas?\b", minuscule)),
    }


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def process(chemin_document: str) -> dict:
    """
    Execute l'Agent 1.

    Accepte un PDF, un document Word ou une presentation : la lecture est
    deleguee a `services/extraction_documents`, qui choisit la strategie
    adaptee au format et bascule sur l'OCR si un PDF n'a pas de couche texte.

    Retourne :
    {
        "nb_pages", "nb_mots", "nb_caracteres", "langue_detectee",
        "methode_extraction", "ocr_utilise", "metadonnees",
        "format_source", "pagination_fiable",
        "chapitres": [{"titre", "extrait", "contenu", "page", "nb_mots"}],
        "mots_cles": [{"mot", "occurrences", "frequence_pct"}],
        "elements_pedagogiques": {...},
        "texte_complet", "texte_brut_tronque"
    }

    Leve `ValueError` avec un message affichable quand le document ne peut pas
    etre lu — l'appelant n'a pas a connaitre le detail des formats.
    """
    try:
        lecture = extraction_documents.lire(chemin_document)
    except DocumentIllisible as exc:
        # Le contrat de l'agent reste inchange : le pipeline attend une
        # ValueError, et le message est deja redige pour l'utilisateur.
        raise ValueError(str(exc)) from exc

    pages = lecture["pages"]
    texte_complet = "\n".join(pages)

    chapitres = _detecter_chapitres(pages, lecture.get("titres"))
    mots_cles = _extraire_mots_cles(texte_complet)

    return {
        "nb_pages": len(pages),
        "nb_mots": len(texte_complet.split()),
        "nb_caracteres": len(texte_complet),
        "langue_detectee": _detecter_langue(texte_complet),
        "methode_extraction": lecture["methode"],
        "ocr_utilise": lecture["ocr_utilise"],
        "metadonnees": lecture["metadonnees"],
        "format_source": lecture["format"],
        # Un .docx n'a pas de pages : le signaler evite que l'annexe
        # d'accreditation presente un numero invente comme une localisation.
        "pagination_fiable": lecture["pagination_fiable"],
        "structure_declaree": bool(lecture.get("titres")),
        "chapitres": chapitres,
        "nb_chapitres": len(chapitres),
        "mots_cles": mots_cles,
        "elements_pedagogiques": _compter_elements_pedagogiques(texte_complet),
        "texte_complet": texte_complet,
        "texte_brut_tronque": texte_complet[:2000],
    }
