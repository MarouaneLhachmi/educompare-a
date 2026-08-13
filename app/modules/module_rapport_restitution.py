"""
Module Rapport et Restitution
==============================

Role (rapport de conception, section 3.3.1) : assurer la mise en forme finale
des resultats d'une analyse, sous la forme d'un rapport structure et d'une
synthese exploitable directement par l'utilisateur, et **en garantir la
fidelite vis-a-vis des donnees effectivement calculees par le systeme**.

Ce module ne produit aucune donnee nouvelle : il ne fait que preparer,
ordonner et mettre en forme ce que les neuf agents ont calcule. Deux sorties :

1. le contexte d'affichage consomme par le template HTML du rapport ;
2. l'export PDF telechargeable, mis en page avec fpdf2.
"""

from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.services import synthese_extractive
from app.services.pedagogie import NIVEAUX_BLOOM

# Palette du document PDF (alignee sur l'identite visuelle de l'interface web).
BLEU_NUIT = (14, 33, 66)
BLEU = (37, 99, 235)
VIOLET = (124, 58, 237)
VERT = (22, 163, 74)
ORANGE = (234, 88, 12)
ROUGE = (220, 38, 38)
GRIS = (108, 117, 125)
GRIS_CLAIR = (241, 245, 249)

COULEURS_STATUT = {
    "Couverte": VERT,
    "Partiellement couverte": ORANGE,
    "Non couverte": ROUGE,
}

COULEURS_PRIORITE = {
    "Critique": ROUGE,
    "Haute": ORANGE,
    "Moyenne": BLEU,
    "Faible": GRIS,
}


# ---------------------------------------------------------------------------
# 1. Contexte d'affichage HTML
# ---------------------------------------------------------------------------

def _couleur_note(note: float) -> str:
    if note >= 80:
        return "excellent"
    if note >= 65:
        return "bon"
    if note >= 50:
        return "moyen"
    if note >= 35:
        return "faible"
    return "critique"


PALIERS_SUIVI = [
    (0.00, "Non abordé"),
    (0.35, "Découverte"),
    (0.55, "Application"),
    (0.75, "Maîtrise"),
    (0.90, "Transfert"),
]


def _palier(maitrise: float) -> str:
    """Situe un niveau de maitrise sur l'echelle de progression affichee."""
    libelle = PALIERS_SUIVI[0][1]
    for seuil, nom in PALIERS_SUIVI:
        if maitrise >= seuil:
            libelle = nom
    return libelle


def _grille_suivi(parcours: dict, contenu_par_rang: dict) -> list[dict]:
    """
    Construit la grille de suivi de l'eleve : une ligne par etape du parcours,
    indiquant le palier de depart, le palier vise et le critere de validation.

    C'est la traduction operationnelle du parcours : ce que l'enseignant
    observe pour decider qu'une etape est acquise et qu'il peut passer a la
    suivante.
    """
    lignes = []
    for etape in parcours.get("etapes", []):
        contenu = contenu_par_rang.get(etape["rang"], {})
        lignes.append({
            "rang": etape["rang"],
            "notion": etape["notion"],
            "pays": etape["pays"],
            "intervention": etape["intervention_nom"],
            "bloom_cible": etape["bloom_cible"],
            "seance": etape["seance_cumulee"],
            "palier_depart": _palier(etape["maitrise_avant"]),
            "palier_vise": _palier(etape["maitrise_apres_predite"]),
            "maitrise_avant": etape["maitrise_avant"],
            "maitrise_visee": etape["maitrise_apres_predite"],
            "critere_reussite": contenu.get("critere_reussite", ""),
            "erreur_frequente": contenu.get("erreur_frequente", ""),
            "prerequis_bloquant": etape.get("prerequis_bloquant", False),
        })
    return lignes


def build_report_context(analyse: dict) -> dict:
    """
    Prepare les donnees pour le template `report.html`.

    Aucune valeur n'est inventee ici : toutes proviennent de `analyse`. Le
    module se limite a du tri, des arrondis, des valeurs par defaut et au
    calcul de grandeurs d'affichage (pourcentages de barres, classes CSS).
    """
    contexte = dict(analyse)

    agent1 = analyse.get("agent1") or {}
    agent2 = analyse.get("agent2") or {}
    agent3 = analyse.get("agent3") or {}
    agent4 = analyse.get("agent4") or {}
    agent5 = analyse.get("agent5") or {}
    agent6 = analyse.get("agent6") or {}
    agent7 = analyse.get("agent7") or {}
    agent8 = analyse.get("agent8") or {}
    agent9 = analyse.get("agent9") or {}

    note = float(agent7.get("note_globale") or 0)
    contexte["note_globale"] = note
    contexte["classe_note"] = _couleur_note(note)
    contexte["niveau_maturite"] = agent7.get("niveau_maturite", "Non évalué")

    # --- Referentiels tries par couverture decroissante ------------------
    pays = sorted(
        (agent6.get("par_pays") or {}).values(),
        key=lambda p: p["taux_couverture_pct"],
        reverse=True,
    )
    # `cle_notion` identifie une notion sans ambiguite entre pays homonymes
    # (« Proportionnalité » existe dans plusieurs referentiels) : c'est la
    # cle utilisee en interne par les agents 5 et 6, et celle sur laquelle
    # s'appuiera la boucle de retour enseignant pour rattacher une etiquette
    # a la notion exacte qu'elle corrige.
    for p in pays:
        for n in p.get("notions", []):
            n.setdefault("cle_notion", f"{n['code']}::{n['notion']}")
    contexte["referentiels"] = pays
    contexte["nb_referentiels"] = len(pays)

    # --- Repartition globale des statuts (pour le graphique en anneau) ---
    total_couvertes = sum(p["nb_couvertes"] for p in pays)
    total_partielles = sum(p["nb_partielles"] for p in pays)
    total_manquantes = sum(p["nb_manquantes"] for p in pays)
    total = max(1, total_couvertes + total_partielles + total_manquantes)
    contexte["repartition"] = {
        "couvertes": total_couvertes,
        "partielles": total_partielles,
        "manquantes": total_manquantes,
        "total": total,
        "pct_couvertes": round(100 * total_couvertes / total, 1),
        "pct_partielles": round(100 * total_partielles / total, 1),
        "pct_manquantes": round(100 * total_manquantes / total, 1),
    }

    # --- Indicateurs d'evaluation (radar + barres) ----------------------
    contexte["contributions"] = agent7.get("contributions", [])
    contexte["indicateurs"] = agent7.get("indicateurs", {})
    contexte["appreciation"] = agent7.get("appreciation", {})
    contexte["clustering"] = agent7.get("clustering", {})
    contexte["modele_evaluation"] = agent7.get("modele", {})

    contexte["notes_par_tete"] = agent7.get("notes_par_tete", {})
    contexte["difficulte"] = agent7.get("difficulte", {})
    contexte["bloom"] = agent7.get("bloom", {})
    contexte["profil_maitrise"] = agent7.get("profil_maitrise", {})

    # Distribution de Bloom agregee sur l'ensemble des chapitres : l'Agent 7
    # ne la produit que chapitre par chapitre.
    distribution = {}
    for profil in (agent7.get("bloom") or {}).get("par_chapitre", {}).values():
        for niveau, nombre in (profil.get("distribution") or {}).items():
            distribution[niveau] = distribution.get(niveau, 0) + nombre
    total_bloom = sum(distribution.values()) or 1
    contexte["bloom_distribution"] = [
        {
            "niveau": niveau,
            "nombre": distribution.get(niveau, 0),
            "part_pct": round(100 * distribution.get(niveau, 0) / total_bloom, 1),
            "atteint": distribution.get(niveau, 0) > 0,
        }
        for niveau in NIVEAUX_BLOOM
    ]

    # Courbe de progression prete a serialiser pour le trace SVG.
    contexte["courbe_parcours"] = [
        {
            "x": point["seance"],
            "y": point["maitrise_ponderee"],
            "libelle": f"Séance {point['seance']:g}",
        }
        for point in (agent8.get("parcours_algorithmique") or {}).get(
            "courbe_progression", []
        )
    ]

    # --- Recommandations -------------------------------------------------
    contexte["recommandations"] = agent8.get("recommandations", [])
    contexte["plan_action"] = agent8.get("plan_action", {})
    contexte["synthese_recommandations"] = agent8.get("synthese", "")

    # --- Plan d'amelioration : les quatre blocs separes de l'Agent 8 -----
    parcours = agent8.get("parcours_algorithmique") or {}
    contenu = agent8.get("contenu_pedagogique") or {}
    contexte["parcours"] = parcours
    contexte["parcours_disponible"] = bool(parcours.get("disponible"))
    contexte["contenu_pedagogique"] = contenu
    contexte["controle_coherence"] = agent8.get("controle_coherence", {})
    contexte["recommandations_gemini"] = agent8.get("recommandations_gemini", {})
    contexte["confrontation"] = agent8.get("confrontation", {})
    contexte["benchmark_rl"] = (parcours.get("modele_apprentissage") or {}).get(
        "evaluation_comparative", {}
    )

    # Contenu pedagogique indexe par rang, pour l'afficher au fil du parcours
    # plutot que dans une liste separee.
    contexte["contenu_par_rang"] = {
        etape.get("rang"): etape for etape in contenu.get("etapes", [])
    }

    # Grille de suivi : chaque etape du parcours devient une ligne de
    # progression avec son palier de depart, son palier vise et son critere.
    contexte["grille_suivi"] = _grille_suivi(parcours, contexte["contenu_par_rang"])

    # --- Notions les moins maitrisees (entree du planificateur) ----------
    contexte["maitrise_faible"] = (agent7.get("profil_maitrise") or {}).get(
        "par_notion", []
    )[:12]

    # --- Synthese extractive, produite sans modele de langage -------------
    # Elle ne remplace pas la synthese executive de l'Agent 9 : elle la
    # double par une methode independante et reproductible. Confronter une
    # synthese generee a une synthese extraite du document lui-meme est le
    # controle le plus simple contre une reformulation qui s'eloignerait du
    # contenu reel.
    try:
        contexte["synthese_extractive"] = synthese_extractive.resumer_analyse(analyse)
    except Exception as exc:
        contexte["synthese_extractive"] = {"disponible": False, "motif": str(exc)[:160]}

    # --- Rapport final ----------------------------------------------------
    contexte["chiffres_cles"] = agent9.get("chiffres_cles", [])
    contexte["synthese_executive"] = agent9.get("synthese_executive", "")
    contexte["verdict"] = agent9.get("verdict", "")
    contexte["message_enseignant"] = agent9.get("message_enseignant", "")
    contexte["fiabilite"] = agent9.get("fiabilite", {})
    contexte["classement"] = agent9.get("classement_referentiels", [])
    contexte["titre_rapport"] = agent9.get("titre_rapport") or analyse.get("nom_fichier", "")

    # --- Traçabilité technique du pipeline --------------------------------
    contexte["pipeline"] = {
        "extraction": agent1.get("methode_extraction"),
        "ocr": agent1.get("ocr_utilise", False),
        "nb_unites": agent3.get("nb_unites", 0),
        "taille_moyenne_unite": agent3.get("taille_moyenne_mots", 0),
        "moteur_vectorisation": agent4.get("moteur"),
        "type_moteur": agent4.get("type_moteur"),
        "dimension": agent4.get("dimension"),
        "index": agent5.get("index", {}),
        "nb_notions_indexees": agent5.get("nb_notions_indexees", 0),
        "seuils": agent6.get("seuils_appliques", {}),
        "duree_s": analyse.get("duree_analyse_s"),
        "reranking": agent6.get("reranking", {}),
        "decision": agent6.get("decision", {}),
        "graphe_prerequis": agent8.get("graphe_prerequis", {}),
    }

    contexte["analyse_qualitative"] = agent6.get("gemini", {})
    contexte["notions_manquantes"] = agent6.get("notions_manquantes", [])[:20]
    contexte["notions_superficielles"] = agent6.get("notions_superficielles", [])[:15]
    contexte["notions_incertaines"] = agent6.get("notions_incertaines", [])[:15]
    contexte["contenus_excedentaires"] = agent6.get("contenus_excedentaires", [])[:12]
    contexte["notions_communes"] = agent6.get("notions_communes", [])
    contexte["score_approfondissement"] = agent6.get("score_approfondissement_pct", 0)
    contexte["score_probabiliste"] = agent6.get("score_global_probabiliste_pct", 0)
    contexte["structure"] = agent2
    contexte["mots_cles"] = agent1.get("mots_cles", [])[:24]
    contexte["chapitres"] = agent1.get("chapitres", [])
    contexte["voisins"] = agent5.get("voisins_par_unite", [])[:15]

    return contexte


# ---------------------------------------------------------------------------
# 2. Export PDF
# ---------------------------------------------------------------------------

def _nettoyer(texte) -> str:
    """
    Les polices « core » de fpdf2 sont limitees au jeu latin-1. On translittere
    les caracteres courants (guillemets typographiques, tirets longs, emojis)
    plutot que d'embarquer une police TTF supplementaire.
    """
    if not isinstance(texte, str):
        texte = str(texte)
    remplacements = {
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "œ": "oe",
        "Œ": "OE", " ": " ", "•": "-", "«": '"', "»": '"',
    }
    for avant, apres in remplacements.items():
        texte = texte.replace(avant, apres)
    return texte.encode("latin-1", "replace").decode("latin-1")


class RapportPDF(FPDF):
    """Document PDF avec en-tete et pied de page personnalises."""

    def __init__(self, analyse: dict):
        super().__init__()
        self.analyse = analyse
        self.set_auto_page_break(auto=True, margin=18)
        self.set_title(_nettoyer(analyse.get("titre_rapport") or "Rapport EduCompare AI"))

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*BLEU_NUIT)
        self.cell(0, 8, "EduCompare AI", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRIS)
        self.cell(0, 8, _nettoyer(self.analyse.get("nom_fichier", "")), align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GRIS_CLAIR)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRIS)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _titre_section(pdf: RapportPDF, numero: str, texte: str) -> None:
    pdf.ln(3)
    pdf.set_fill_color(*BLEU_NUIT)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, _nettoyer(f"  {numero}. {texte}"), fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def _sous_titre(pdf: RapportPDF, texte: str) -> None:
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*BLEU)
    pdf.multi_cell(0, 6, _nettoyer(texte), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _paragraphe(pdf: RapportPDF, texte: str, taille: float = 9.5) -> None:
    if not texte:
        return
    pdf.set_font("Helvetica", "", taille)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5.2, _nettoyer(texte), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.5)


def _puces(pdf: RapportPDF, elements, taille: float = 9.5) -> None:
    pdf.set_font("Helvetica", "", taille)
    pdf.set_text_color(30, 30, 30)
    for element in elements:
        pdf.multi_cell(0, 5, _nettoyer(f"  -  {element}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.5)


def _barre(pdf: RapportPDF, x: float, y: float, largeur: float, hauteur: float,
           pourcentage: float, couleur) -> None:
    """Dessine une barre de progression horizontale."""
    pdf.set_fill_color(*GRIS_CLAIR)
    pdf.rect(x, y, largeur, hauteur, style="F")
    remplissage = max(0.0, min(100.0, float(pourcentage))) / 100.0 * largeur
    if remplissage > 0:
        pdf.set_fill_color(*couleur)
        pdf.rect(x, y, remplissage, hauteur, style="F")


def _courbe(pdf: RapportPDF, points: list[dict], hauteur: float = 34.0) -> None:
    """Trace la courbe de progression de la maitrise, seance apres seance."""
    if len(points) < 2:
        return

    x0, largeur = 20.0, 168.0
    y0 = pdf.get_y() + 2
    valeurs = [p["maitrise_ponderee"] for p in points]
    v_min, v_max = min(valeurs), max(valeurs)
    # Echelle resserree autour des valeurs observees : sur un gain de quelques
    # centiemes, un axe 0-1 aplatirait completement la courbe.
    marge = max(0.03, (v_max - v_min) * 0.35)
    bas, haut = max(0.0, v_min - marge), min(1.0, v_max + marge)
    amplitude = (haut - bas) or 1.0

    # Cadre et graduations
    pdf.set_draw_color(*GRIS_CLAIR)
    pdf.set_line_width(0.2)
    for i in range(3):
        valeur = bas + amplitude * i / 2
        y = y0 + hauteur * (1 - (valeur - bas) / amplitude)
        pdf.line(x0, y, x0 + largeur, y)
        pdf.set_xy(x0 - 12, y - 2)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_text_color(*GRIS)
        pdf.cell(11, 4, f"{valeur * 100:.0f}%", align="R")

    # Trace
    pdf.set_draw_color(*BLEU)
    pdf.set_line_width(0.7)
    coordonnees = [
        (
            x0 + largeur * i / (len(points) - 1),
            y0 + hauteur * (1 - (p["maitrise_ponderee"] - bas) / amplitude),
        )
        for i, p in enumerate(points)
    ]
    for (xa, ya), (xb, yb) in zip(coordonnees, coordonnees[1:]):
        pdf.line(xa, ya, xb, yb)

    pdf.set_fill_color(*BLEU)
    for x, y in coordonnees:
        pdf.circle(x=x - 0.7, y=y - 0.7, radius=1.4, style="F")

    pdf.set_line_width(0.2)
    pdf.set_xy(x0, y0 + hauteur + 1)
    pdf.set_font("Helvetica", "", 6.5)
    pdf.set_text_color(*GRIS)
    pdf.cell(largeur / 2, 4, "Depart")
    pdf.cell(largeur / 2, 4, f"Seance {points[-1]['seance']:g}", align="R",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _page_de_garde(pdf: RapportPDF, analyse: dict) -> None:
    agent7 = analyse.get("agent7") or {}
    agent9 = analyse.get("agent9") or {}
    note = float(agent7.get("note_globale") or 0)

    pdf.add_page()
    # Bandeau
    pdf.set_fill_color(*BLEU_NUIT)
    pdf.rect(0, 0, 210, 62, style="F")
    pdf.set_xy(14, 16)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "EduCompare AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(190, 205, 230)
    pdf.cell(0, 7, "Rapport d'analyse comparative de support de cours",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _nettoyer(
        f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')} - reference {analyse.get('id')}"
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(72)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 7, _nettoyer(agent9.get("titre_rapport") or analyse.get("nom_fichier", "")),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(0, 5, _nettoyer(
        f"Fichier : {analyse.get('nom_fichier')}   |   Matiere : {analyse.get('matiere')}   |   "
        f"Niveau : {analyse.get('niveau')}\n"
        f"Depose par : {analyse.get('utilisateur_nom') or 'session de demonstration'}   |   "
        f"Date : {analyse.get('date_creation')}"
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Encadre de la note globale
    y = pdf.get_y()
    couleur = VERT if note >= 65 else (ORANGE if note >= 45 else ROUGE)
    pdf.set_fill_color(*GRIS_CLAIR)
    pdf.rect(14, y, 182, 30, style="F")
    pdf.set_xy(20, y + 5)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*couleur)
    pdf.cell(38, 12, f"{note:.0f}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRIS)
    pdf.set_xy(20, y + 18)
    pdf.cell(38, 6, "/ 100", new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_xy(60, y + 6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*BLEU_NUIT)
    pdf.cell(0, 6, _nettoyer(f"Niveau d'alignement : {agent7.get('niveau_maturite', 'n/a')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_xy(60, y + 14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(130, 4.6, _nettoyer(agent9.get("verdict") or agent7.get("message_maturite", "")),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(y + 36)

    # Chiffres cles
    chiffres = (agent9.get("chiffres_cles") or [])[:6]
    if chiffres:
        _sous_titre(pdf, "Chiffres cles")
        depart_y = pdf.get_y()
        for index, chiffre in enumerate(chiffres):
            colonne = index % 3
            ligne = index // 3
            x = 14 + colonne * 61
            y_case = depart_y + ligne * 20
            pdf.set_fill_color(248, 250, 252)
            pdf.rect(x, y_case, 57, 17, style="F")
            pdf.set_xy(x + 3, y_case + 2.5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*BLEU)
            pdf.cell(50, 6, _nettoyer(f"{chiffre['valeur']}{chiffre.get('unite', '')}"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_xy(x + 3, y_case + 9.5)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*GRIS)
            pdf.multi_cell(51, 3.4, _nettoyer(chiffre["libelle"]))
        pdf.set_y(depart_y + ((len(chiffres) - 1) // 3 + 1) * 20 + 3)

    # Synthese extractive : produite sans modele de langage, elle sert de
    # contrepoint verifiable a la synthese generee.
    try:
        extractive = synthese_extractive.resumer_analyse(analyse, nb_phrases=4)
    except Exception:
        extractive = {"disponible": False}
    if extractive.get("disponible"):
        _sous_titre(pdf, "Ce que dit le document lui-meme")
        _paragraphe(pdf, extractive["resume"], taille=9)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 4, _nettoyer(
            f"Phrases extraites du document par {extractive['algorithme']} — "
            f"{extractive['nb_phrases_retenues']} retenues sur "
            f"{extractive['nb_phrases_source']}. Aucune reformulation, aucun "
            f"modele de langage."
        ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # Synthese executive
    if agent9.get("synthese_executive"):
        _sous_titre(pdf, "Synthese executive")
        _paragraphe(pdf, agent9["synthese_executive"])
        if not agent9.get("synthese_generee_par_ia"):
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*ORANGE)
            pdf.multi_cell(0, 4, _nettoyer(
                "Synthese produite par gabarit deterministe : le modele de langage n'etait pas disponible."
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)


def export_pdf(analyse: dict, output_path: str) -> str:
    """Genere le rapport PDF complet. Retourne le chemin du fichier produit."""
    agent1 = analyse.get("agent1") or {}
    agent2 = analyse.get("agent2") or {}
    agent3 = analyse.get("agent3") or {}
    agent4 = analyse.get("agent4") or {}
    agent5 = analyse.get("agent5") or {}
    agent6 = analyse.get("agent6") or {}
    agent7 = analyse.get("agent7") or {}
    agent8 = analyse.get("agent8") or {}
    agent9 = analyse.get("agent9") or {}

    pdf = RapportPDF(analyse)
    pdf.alias_nb_pages()
    _page_de_garde(pdf, analyse)

    # ------------------------------------------------------------------
    # 1. Structure du cours
    # ------------------------------------------------------------------
    pdf.add_page()
    _titre_section(pdf, "1", "Structure du cours (Agents 1 a 3)")
    _paragraphe(pdf, (
        f"Pages : {agent1.get('nb_pages', 0)}   |   Mots : {agent1.get('nb_mots', 0)}   |   "
        f"Langue detectee : {agent1.get('langue_detectee', '-')}   |   "
        f"Extraction : {agent1.get('methode_extraction', '-')}"
    ))
    if agent2.get("resume"):
        _sous_titre(pdf, "Lecture pedagogique du document (Agent 2)")
        _paragraphe(pdf, agent2["resume"])
        _paragraphe(pdf, (
            f"Discipline identifiee : {agent2.get('discipline_identifiee', '-')}   |   "
            f"Niveau estime : {agent2.get('niveau_estime', '-')}"
        ))

    _sous_titre(pdf, f"Chapitres detectes ({len(agent1.get('chapitres', []))})")
    for chapitre in agent1.get("chapitres", [])[:20]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, _nettoyer(f"  {chapitre['titre']}  (p. {chapitre.get('page', '-')})"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 4.2, _nettoyer("     " + (chapitre.get("extrait", "")[:220])),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    _paragraphe(pdf, (
        f"Decoupage (Agent 3) : {agent3.get('nb_unites', 0)} unites de sens, "
        f"taille moyenne {agent3.get('taille_moyenne_mots', 0)} mots, "
        f"strategie « {agent3.get('strategie', '-')} »."
    ))
    if agent1.get("mots_cles"):
        _paragraphe(pdf, "Mots-cles dominants : " + ", ".join(
            m["mot"] for m in agent1["mots_cles"][:18]
        ))

    # ------------------------------------------------------------------
    # 2. Comparaison
    # ------------------------------------------------------------------
    pdf.add_page()
    _titre_section(pdf, "2", "Comparaison aux referentiels etrangers (Agents 4 a 6)")
    reranking_infos = agent6.get("reranking") or {}
    _paragraphe(pdf, (
        f"Vectorisation : {agent4.get('moteur', '-')} ({agent4.get('dimension', 0)} dimensions).  "
        f"Base vectorielle : {(agent5.get('index') or {}).get('moteur', '-')}, "
        f"{agent5.get('nb_notions_indexees', 0)} notions indexees.  "
        + (
            f"Re-ranking : cross-encodeur sur {reranking_infos.get('nb_paires_scorees', 0)} paires."
            if reranking_infos.get("applique") else "Re-ranking : non applique."
        )
    ))
    _paragraphe(pdf, (
        f"Couverture globale {agent6.get('score_global_pct', 0)} %, dont "
        f"{agent6.get('score_approfondissement_pct', 0)} % de notions traitees avec assez de "
        f"matiere pour etre reellement apprises. "
        f"{agent6.get('nb_notions_manquantes', 0)} notion(s) absente(s), "
        f"{agent6.get('nb_notions_superficielles', 0)} traitee(s) trop brievement, "
        f"{agent6.get('nb_notions_incertaines', 0)} en zone d'incertitude."
    ), taille=9)

    for pays in sorted((agent6.get("par_pays") or {}).values(),
                       key=lambda p: p["taux_couverture_pct"], reverse=True):
        pdf.ln(1)
        _sous_titre(pdf, f"{pays['pays']} - {pays.get('referentiel', '')}")
        y = pdf.get_y()
        taux = pays["taux_couverture_pct"]
        couleur = VERT if taux >= 70 else (ORANGE if taux >= 45 else ROUGE)
        _barre(pdf, 14, y, 150, 5, taux, couleur)
        pdf.set_xy(168, y - 1.2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*couleur)
        pdf.cell(28, 6, f"{taux} %", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        _paragraphe(pdf, (
            f"{pays['nb_couvertes']} notion(s) couverte(s), {pays['nb_partielles']} partielle(s), "
            f"{pays['nb_manquantes']} manquante(s) sur {pays['nb_notions']} - "
            f"similarite moyenne {pays['score_similarite_moyen']}."
        ), taille=9)

        # Tableau des notions : la probabilite de couverture repond a « la
        # notion est-elle abordee ? », la matiere a « y a-t-il assez de contenu
        # pour qu'elle soit apprise ? ». Les deux sont mesurees separement.
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(*GRIS_CLAIR)
        pdf.cell(92, 6, "  Notion du referentiel", fill=True)
        pdf.cell(22, 6, "Abordee", fill=True, align="C")
        pdf.cell(22, 6, "Matiere", fill=True, align="C")
        pdf.cell(0, 6, "Diagnostic", fill=True, align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        for notion in pays["notions"]:
            if pdf.get_y() > 258:
                pdf.add_page()
            marque = " ?" if notion.get("incertaine") else ""
            pdf.cell(92, 5.4, _nettoyer("  " + notion["notion"][:56] + marque))
            pdf.cell(22, 5.4, f"{notion.get('probabilite_couverture', 0) * 100:.0f} %", align="C")
            pdf.cell(22, 5.4, f"{notion.get('suffisance', 0) * 100:.0f} %", align="C")
            pdf.set_text_color(*COULEURS_STATUT.get(notion["statut"], GRIS))
            pdf.cell(0, 5.4, _nettoyer(notion.get("libelle_ecart", notion["statut"])),
                     align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 3.6, _nettoyer(
            "  « ? » signale une notion en zone d'incertitude : le verdict automatique n'y est "
            "pas fiable et demande une relecture."
        ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    qualitative = agent6.get("gemini") or {}
    if qualitative.get("disponible"):
        pdf.add_page()
        _sous_titre(pdf, "Lecture qualitative de la comparaison")
        _paragraphe(pdf, qualitative.get("synthese_comparative", ""))
        if qualitative.get("analyse_par_pays"):
            for entree in qualitative["analyse_par_pays"]:
                _paragraphe(pdf, f"{entree.get('pays', '')} : {entree.get('lecture', '')}", taille=9)
        if qualitative.get("points_forts"):
            _sous_titre(pdf, "Points forts identifies")
            _puces(pdf, qualitative["points_forts"])
        if qualitative.get("ecarts_majeurs"):
            _sous_titre(pdf, "Ecarts majeurs")
            _puces(pdf, [
                f"{e.get('notion', '')} ({e.get('pays', '')}) : {e.get('impact', '')}"
                for e in qualitative["ecarts_majeurs"]
            ])
        if qualitative.get("specificites_locales"):
            _sous_titre(pdf, "Specificites locales a valoriser")
            _puces(pdf, qualitative["specificites_locales"])
    else:
        _paragraphe(pdf, (
            "[Analyse qualitative indisponible] La cartographie chiffree ci-dessus reste "
            "integralement valide ; seule la lecture redigee par le modele de langage n'a "
            "pas pu etre produite."
        ))

    # ------------------------------------------------------------------
    # 3. Evaluation
    # ------------------------------------------------------------------
    pdf.add_page()
    _titre_section(pdf, "3", "Evaluation pedagogique (Agent 7)")
    tetes = agent7.get("notes_par_tete") or {}
    _paragraphe(pdf, (
        f"Note globale : {agent7.get('note_globale', 0)}/100 - niveau « "
        f"{agent7.get('niveau_maturite', '-')} ». "
        f"Modele : {(agent7.get('modele') or {}).get('algorithme', '-')}."
        + (
            f" Les deux tetes de l'ensemble predisent {tetes['gradient_boosting']} "
            f"(arbres boostes) et {tetes['reseau_de_neurones']} (reseau de neurones) : "
            f"ecart {tetes['divergence']}, fiabilite {tetes['fiabilite']}."
            if tetes else ""
        )
    ))

    for contribution in agent7.get("contributions", []):
        y = pdf.get_y()
        if y > 250:
            pdf.add_page()
            y = pdf.get_y()
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(88, 6, _nettoyer(contribution["libelle"][:52]))
        valeur = contribution["valeur"] * 100
        couleur = VERT if valeur >= 66 else (ORANGE if valeur >= 40 else ROUGE)
        _barre(pdf, 105, y + 1.5, 60, 4, valeur, couleur)
        pdf.set_xy(168, y)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(28, 6, f"{contribution['valeur']:.2f}  ({contribution['poids_pct']:.0f} %)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Difficulte, progression et profondeur cognitive
    difficulte = agent7.get("difficulte") or {}
    bloom = agent7.get("bloom") or {}
    if difficulte.get("mediane") is not None:
        progression = difficulte.get("progression") or {}
        bande = difficulte.get("bande_attendue") or [0, 0]
        _paragraphe(pdf, (
            f"Difficulte du texte : mediane {difficulte['mediane']} pour une bande attendue de "
            f"{bande[0]} a {bande[1]} au cycle « {difficulte.get('cycle', '-')} » "
            f"(ecart {difficulte.get('ecart_au_cycle', 0)}). "
            f"Progression : tau de Kendall {progression.get('tau_kendall', 0)}, "
            f"{len(progression.get('ruptures', []))} rupture(s) de difficulte. "
            f"Methode : {str(difficulte.get('source', '')).replace('_', ' ')}."
        ), taille=8.5)
    if bloom.get("niveau_max_atteint"):
        _paragraphe(pdf, (
            f"Profondeur cognitive (taxonomie de Bloom) : niveau maximal atteint "
            f"« {bloom['niveau_max_atteint']} », profondeur moyenne "
            f"{bloom.get('profondeur_moyenne', 0)}. Un support qui plafonne a « Memoriser » "
            f"ou « Comprendre » fait restituer, il ne fait ni analyser ni creer."
        ), taille=8.5)

    profil = agent7.get("profil_maitrise") or {}
    if profil.get("par_notion"):
        _paragraphe(pdf, (
            f"Profil de maitrise : {profil['nb_notions']} notions evaluees, maitrise moyenne "
            f"{profil['maitrise_globale']}, dont {profil['nb_maitrise_faible']} faible(s) et "
            f"{profil['nb_maitrise_solide']} solide(s). {profil.get('avertissement', '')}"
        ), taille=8.5)

    appreciation = agent7.get("appreciation") or {}
    if appreciation.get("appreciation_globale"):
        _sous_titre(pdf, "Appreciation de l'evaluateur")
        _paragraphe(pdf, appreciation["appreciation_globale"])
    if appreciation.get("levier_prioritaire"):
        _paragraphe(pdf, "Levier prioritaire : " + appreciation["levier_prioritaire"], taille=9)
    if appreciation.get("risques_accreditation"):
        _sous_titre(pdf, "Risques au regard de l'accreditation")
        _puces(pdf, appreciation["risques_accreditation"])
    if appreciation.get("atouts_a_valoriser"):
        _sous_titre(pdf, "Atouts a valoriser")
        _puces(pdf, appreciation["atouts_a_valoriser"])

    clustering = agent7.get("clustering") or {}
    if clustering.get("applique"):
        _paragraphe(pdf, (
            f"Coherence thematique mesuree par regroupement non supervise "
            f"({clustering.get('algorithme', '-')}) : {clustering.get('nb_groupes_optimal', '-')} "
            f"groupes thematiques, silhouette {clustering.get('silhouette', '-')}."
        ), taille=8.5)

    # ------------------------------------------------------------------
    # 4. Plan d'amelioration — bloc algorithmique (Agent 8)
    # ------------------------------------------------------------------
    parcours = agent8.get("parcours_algorithmique") or {}
    contenu = agent8.get("contenu_pedagogique") or {}
    contenu_par_rang = {e.get("rang"): e for e in contenu.get("etapes", [])}

    pdf.add_page()
    _titre_section(pdf, "4", "Plan d'amelioration - parcours planifie (Agent 8)")

    if parcours.get("disponible"):
        initial = parcours["etat_initial"]["maitrise_ponderee"] * 100
        final = parcours["etat_final_predit"]["maitrise_ponderee"] * 100
        retenu = parcours["etat_final_predit"]["retention_ponderee"] * 100

        _paragraphe(pdf, (
            f"{parcours['nb_etapes']} etapes reparties sur {parcours['seances_planifiees']} seances, "
            f"couvrant {parcours['nb_notions_distinctes']} notions distinctes. "
            f"Maitrise ponderee attendue : {initial:.1f} % -> {final:.1f} %, "
            f"dont {retenu:.1f} % encore acquis apres huit periodes sans pratique."
        ))
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 4.2, _nettoyer(
            "Sequence decidee par apprentissage par renforcement a partir du profil de maitrise "
            "mesure, du graphe de prerequis et de l'importance internationale de chaque notion. "
            "Aucun texte genere n'intervient dans ces choix."
        ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        _sous_titre(pdf, "Trajectoire de maitrise prevue")
        _courbe(pdf, parcours.get("courbe_progression", []))

        _sous_titre(pdf, "Deroule seance par seance")
        for etape in parcours.get("etapes", []):
            if pdf.get_y() > 235:
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_fill_color(*BLEU)
            pdf.rect(14, y, 2.5, 17, style="F")

            pdf.set_xy(19, y)
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*BLEU_NUIT)
            titre_etape = (contenu_par_rang.get(etape["rang"], {}).get("titre_seance")
                           or etape["intervention_nom"])
            pdf.multi_cell(176, 4.8, _nettoyer(
                f"{etape['rang']}. Seance {etape['seance_cumulee']:g} - {titre_etape}"
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_x(19)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*BLEU)
            pdf.multi_cell(176, 4.2, _nettoyer(
                f"{etape['intervention_nom']}  |  {etape['notion']} ({etape['pays']})  |  "
                f"Bloom : {etape['bloom_cible']}  |  "
                f"maitrise {etape['maitrise_avant']:.2f} -> {etape['maitrise_apres_predite']:.2f}"
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_x(19)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(176, 4, _nettoyer(etape["justification"]),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        # --- Grille de suivi ------------------------------------------
        pdf.add_page()
        _sous_titre(pdf, "Grille de suivi de l'eleve")
        _paragraphe(pdf, (
            "Une ligne par etape : le palier vise et le critere permettant de considerer "
            "l'etape acquise avant de passer a la suivante."
        ), taille=8.5)

        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_fill_color(*GRIS_CLAIR)
        pdf.cell(14, 6, "  Sce", fill=True)
        pdf.cell(62, 6, "Notion", fill=True)
        pdf.cell(40, 6, "Activite", fill=True)
        pdf.cell(30, 6, "Palier vise", fill=True, align="C")
        pdf.cell(40, 6, "Critere", fill=True)
        pdf.cell(0, 6, "Acquis", fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", "", 7.5)
        for ligne in _grille_suivi(parcours, contenu_par_rang):
            if pdf.get_y() > 262:
                pdf.add_page()
            pdf.cell(14, 5.4, _nettoyer(f"  {ligne['seance']:g}"))
            pdf.cell(62, 5.4, _nettoyer(ligne["notion"][:40]))
            pdf.cell(40, 5.4, _nettoyer(ligne["intervention"][:26]))
            pdf.cell(30, 5.4, _nettoyer(ligne["palier_vise"]), align="C")
            pdf.cell(40, 5.4, _nettoyer((ligne["critere_reussite"] or "-")[:26]))
            pdf.cell(0, 5.4, "[  ]", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
    else:
        _paragraphe(pdf, "Le parcours d'amelioration n'a pas pu etre planifie pour cette analyse.")

    # ------------------------------------------------------------------
    # 5. Contenu pedagogique redige
    # ------------------------------------------------------------------
    if contenu.get("etapes"):
        pdf.add_page()
        _titre_section(pdf, "5", "Contenu pedagogique des premieres etapes")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*GRIS)
        pdf.multi_cell(0, 4.2, _nettoyer(
            "Theorie et exercices rediges par le modele de langage POUR les etapes decidees "
            "ci-dessus. Le modele n'a choisi ni les notions ni leur ordre : il en redige le "
            "contenu. A relire avant utilisation en classe."
            if contenu.get("disponible") else
            "Contenu produit par gabarit deterministe : le modele de langage n'etait pas disponible."
        ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        for etape in contenu["etapes"]:
            if pdf.get_y() > 215:
                pdf.add_page()
            _sous_titre(pdf, f"Etape {etape['rang']} - {etape.get('titre_seance', '')}")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GRIS)
            pdf.multi_cell(0, 4.2, _nettoyer(
                f"Notion : {etape['notion']}  |  {etape['intervention_nom']}  |  "
                f"Niveau de Bloom vise : {etape['bloom_cible']}"
            ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

            theorie = etape.get("theorie") or {}
            if theorie.get("rappel"):
                _paragraphe(pdf, "Prerequis a reactiver : " + theorie["rappel"], taille=8.5)
            if theorie.get("apport"):
                _paragraphe(pdf, theorie["apport"], taille=9)
            if theorie.get("exemple"):
                _paragraphe(pdf, "Exemple file : " + theorie["exemple"], taille=8.5)

            for exercice in etape.get("exercices", []):
                if pdf.get_y() > 258:
                    pdf.add_page()
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*BLEU)
                pdf.multi_cell(0, 4.2, _nettoyer(f"  [{exercice['niveau']}]"),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 8.5)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 4.3, _nettoyer("  " + exercice["enonce"]),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                if exercice.get("reponse"):
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(*GRIS)
                    pdf.multi_cell(0, 4.2, _nettoyer("  Reponse : " + exercice["reponse"]),
                                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0)

            if etape.get("critere_reussite"):
                _paragraphe(pdf, "Critere de reussite : " + etape["critere_reussite"], taille=8.5)
            if etape.get("erreur_frequente"):
                _paragraphe(pdf, "Erreur frequente : " + etape["erreur_frequente"], taille=8.5)
            pdf.ln(2)

        controle = agent8.get("controle_coherence") or {}
        if controle.get("applique"):
            _paragraphe(pdf, (
                f"Controle automatique des enonces : {controle['part_correctement_cibles_pct']} % "
                f"des {controle['nb_exercices_verifies']} exercices generes sont plus proches de "
                f"leur notion cible que de toute autre notion du parcours. {controle['verdict']}"
            ), taille=8.5)

    # ------------------------------------------------------------------
    # 6. Les deux sources
    # ------------------------------------------------------------------
    libres = agent8.get("recommandations_gemini") or {}
    confrontation = agent8.get("confrontation") or {}

    pdf.add_page()
    _titre_section(pdf, "6", "Les deux sources de recommandations")
    _paragraphe(pdf, (
        "Les recommandations proviennent de deux origines deliberement separees : le parcours "
        "planifie par nos modeles (section 4) et l'avis libre du modele de langage ci-dessous. "
        "Les confronter permet de distinguer ce qui est solide de ce qui merite un arbitrage."
    ), taille=9)

    if libres.get("disponible"):
        _sous_titre(pdf, "Avis libre du modele de langage")
        if libres.get("synthese"):
            _paragraphe(pdf, libres["synthese"], taille=9)
        for reco in libres.get("recommandations", []):
            if pdf.get_y() > 245:
                pdf.add_page()
            couleur = COULEURS_PRIORITE.get(reco["priorite"].capitalize(), GRIS)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*couleur)
            pdf.multi_cell(0, 4.6, _nettoyer(f"- [{reco['priorite']}] {reco['titre']}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 4.2, _nettoyer("   " + reco.get("argument", "")),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if reco.get("mise_en_oeuvre"):
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(*GRIS)
                pdf.multi_cell(0, 4.2, _nettoyer("   Mise en oeuvre : " + reco["mise_en_oeuvre"]),
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)

        if libres.get("angle_mort"):
            _paragraphe(pdf, "Angle mort signale : " + libres["angle_mort"], taille=8.5)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*ORANGE)
        pdf.multi_cell(0, 4, _nettoyer(libres.get("avertissement", "")),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
    else:
        _paragraphe(pdf, "[Avis libre indisponible] Le modele de langage n'a pas pu etre sollicite.")

    if confrontation.get("disponible"):
        _sous_titre(pdf, f"Confrontation - {confrontation['taux_convergence_pct']} % de convergence")
        _paragraphe(pdf, confrontation.get("lecture", ""), taille=9)
        if confrontation.get("convergences"):
            _puces(pdf, [
                f"{c['notion']} : {c['intervention_algorithmique']} (nos modeles) "
                f"<-> {c['recommandation_gemini']} (modele de langage)"
                for c in confrontation["convergences"]
            ], taille=8.5)
        if confrontation.get("specifiques_gemini"):
            _paragraphe(pdf, "Proposes uniquement par le modele de langage :", taille=8.5)
            _puces(pdf, [
                f"{s['titre']} ({s['notion_visee']})"
                for s in confrontation["specifiques_gemini"][:6]
            ], taille=8.5)

    # ------------------------------------------------------------------
    # 7. Tracabilite
    # ------------------------------------------------------------------
    pdf.add_page()
    _titre_section(pdf, "7", "Tracabilite et fiabilite de l'analyse")
    fiabilite = agent9.get("fiabilite") or {}
    _paragraphe(pdf, (
        f"Niveau de confiance estime : {fiabilite.get('niveau_confiance_pct', 0)} %.  "
        f"Duree totale du traitement : {analyse.get('duree_analyse_s', '-')} s."
    ))
    if fiabilite.get("alertes"):
        _sous_titre(pdf, "Points de vigilance")
        _puces(pdf, fiabilite["alertes"])
    else:
        _paragraphe(pdf, "Aucun repli n'a ete active : tous les agents ont abouti nominalement.")

    _sous_titre(pdf, "Chaine de traitement executee")
    _puces(pdf, [
        f"Agent 1 - Extraction : {agent1.get('methode_extraction', '-')}",
        f"Agent 2 - Comprehension : {'modele de langage' if agent2.get('source') == 'gemini' else 'repli deterministe'}",
        f"Agent 3 - Decoupage : {agent3.get('nb_unites', 0)} unites",
        f"Agent 4 - Vectorisation : {agent4.get('moteur', '-')}",
        f"Agent 5 - Recherche : {(agent5.get('index') or {}).get('moteur', '-')}",
        f"Agent 6 - Comparaison : decision {str((agent6.get('decision') or {}).get('source', '-')).replace('_', ' ')}",
        f"Agent 7 - Evaluation : {(agent7.get('modele') or {}).get('algorithme', '-')}",
        f"Agent 8 - Recommandations : {(parcours.get('moteur') or 'priorisation deterministe')}",
        "Agent 9 - Rapport final : agregation et synthese executive",
    ], taille=8.5)

    # Comparaison du planificateur aux politiques de reference : le chiffre
    # est recalcule a chaque entrainement, il n'est pas declaratif.
    benchmark = (parcours.get("modele_apprentissage") or {}).get("evaluation_comparative") or {}
    if benchmark.get("resultats"):
        _sous_titre(pdf, "Performance du planificateur")
        _paragraphe(pdf, benchmark.get("protocole", ""), taille=8.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(*GRIS_CLAIR)
        pdf.cell(46, 6, "  Politique", fill=True)
        pdf.cell(36, 6, "Recompense", fill=True, align="C")
        pdf.cell(36, 6, "Retention", fill=True, align="C")
        pdf.cell(36, 6, "Maitrise", fill=True, align="C")
        pdf.cell(0, 6, "Consolidation", fill=True, align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        for nom, resultats in benchmark["resultats"].items():
            pdf.cell(46, 5.4, _nettoyer("  " + nom.capitalize()))
            pdf.cell(36, 5.4, f"{resultats['recompense']}", align="C")
            pdf.cell(36, 5.4, f"{resultats['retention_ponderee']}", align="C")
            pdf.cell(36, 5.4, f"{resultats['maitrise_ponderee']}", align="C")
            pdf.cell(0, 5.4, f"{resultats['stabilite']}", align="C",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        gains = benchmark.get("gains_vs_gloutonne_pct") or {}
        _paragraphe(pdf, "Gains de la politique apprise sur la politique gloutonne : " + ", ".join(
            f"{cle.replace('_', ' ')} {valeur:+g} %" for cle, valeur in gains.items()
        ), taille=8.5)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(*GRIS)
    pdf.multi_cell(0, 4, _nettoyer(
        "Rapport genere automatiquement par EduCompare AI. Les scores de similarite sont issus "
        "d'une comparaison semantique automatisee et constituent une aide a la decision : ils ne "
        "se substituent pas a l'expertise pedagogique d'un enseignant ou d'un comite d'accreditation."
    ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_path)
    return output_path
