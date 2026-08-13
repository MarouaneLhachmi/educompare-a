"""
Genere un PDF d'exemple : un faux cours de mathematiques pour la derniere
annee du primaire (6eme AEP au Maroc / equivalent CM2), pratique pour tester
le pipeline sans avoir a chercher un vrai document le jour de la demo.

Usage :
    python create_sample_course.py
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

CHAPITRES = [
    (
        "Chapitre 1 : Les grands nombres",
        "Dans ce chapitre, l'eleve apprend a lire, ecrire et comparer des nombres "
        "entiers jusqu'au million. On travaille la decomposition en unites, dizaines, "
        "centaines, milliers, et la valeur de position de chaque chiffre. Des exercices "
        "de rangement croissant et decroissant sont proposes.",
    ),
    (
        "Chapitre 2 : Les fractions",
        "Introduction aux fractions simples (demis, tiers, quarts), comparaison de "
        "fractions de meme denominateur, et premiere approche de l'addition de "
        "fractions simples. Des exercices visuels avec des figures partagees sont "
        "utilises pour faciliter la comprehension.",
    ),
    (
        "Chapitre 3 : Multiplication et division",
        "Revision des tables de multiplication, technique de la multiplication posee "
        "a deux chiffres, et introduction de la division euclidienne avec reste. "
        "Applications a des problemes concrets de partage et de groupement.",
    ),
    (
        "Chapitre 4 : Geometrie - figures planes",
        "Reconnaissance et proprietes du carre, du rectangle et du triangle. Calcul "
        "du perimetre des figures usuelles. Utilisation de la regle et de l'equerre "
        "pour tracer des figures precises.",
    ),
    (
        "Chapitre 5 : Mesures et grandeurs",
        "Unites de longueur (mm, cm, m, km), de masse (g, kg) et de duree (minute, "
        "heure, jour). Conversion entre unites et resolution de problemes de la vie "
        "courante impliquant des mesures.",
    ),
    (
        "Chapitre 6 : Resolution de problemes",
        "Problemes a plusieurs etapes combinant les quatre operations. Accent mis sur "
        "la comprehension de l'enonce, le choix de la demarche, et la verification du "
        "resultat obtenu.",
    ),
]

OUTPUT_DIR = "sample_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cours_maths_primaire_exemple.pdf")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "Cours de Mathematiques - Derniere annee du primaire",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Document d'exemple genere pour la demonstration du prototype EduCompare AI.",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    for titre, contenu in CHAPITRES:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, titre, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, contenu, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    pdf.output(OUTPUT_FILE)
    print(f"Fichier cree : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
