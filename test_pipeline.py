"""
Script de test en ligne de commande du pipeline complet (Agents 1 a 9).

Utile pour verifier la chaine de traitement AVANT de lancer l'application
Flask, ou pour deboguer un agent en particulier sans passer par le navigateur.

Usage :
    python test_pipeline.py sample_data/cours_maths_primaire_exemple.pdf
    python test_pipeline.py mon_cours.pdf --matiere Sciences --json resultat.json
"""

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from app.modules import module_traitement_analyse  # noqa: E402
from app.services import referentiels  # noqa: E402


def _entete(titre: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {titre}")
    print("=" * 74)


def main() -> int:
    parseur = argparse.ArgumentParser(description="Test du pipeline EduCompare AI")
    parseur.add_argument("pdf", help="chemin du fichier PDF a analyser")
    parseur.add_argument("--matiere", default="Mathématiques")
    parseur.add_argument("--niveau", default="Dernière année du primaire")
    parseur.add_argument("--pays", nargs="*", default=None,
                         help="codes pays a comparer (ex : FR UK US) ; tous par defaut")
    parseur.add_argument("--json", dest="sortie_json", default=None,
                         help="chemin d'export du resultat complet au format JSON")
    args = parseur.parse_args()

    print(f"Analyse de : {args.pdf}")
    print(f"Referentiel : {referentiels.cle_pour(args.matiere, args.niveau)}")

    analyse = module_traitement_analyse.run_analysis(
        pdf_path=args.pdf,
        matiere=args.matiere,
        niveau=args.niveau,
        nom_fichier_original=args.pdf.split("/")[-1].split("\\")[-1],
        pays_reference=args.pays,
    )

    if analyse["statut"] == "ECHEC":
        print(f"\n[ECHEC] {analyse['erreur']}")
        return 1

    agent1 = analyse["agent1"]
    agent2 = analyse["agent2"]
    agent3 = analyse["agent3"]
    agent4 = analyse["agent4"]
    agent5 = analyse["agent5"]
    agent6 = analyse["agent6"]
    agent7 = analyse["agent7"]
    agent8 = analyse["agent8"]
    agent9 = analyse["agent9"]

    _entete("Agents 1 a 3 — Extraction, comprehension, decoupage")
    print(f"Pages           : {agent1['nb_pages']}   Mots : {agent1['nb_mots']}")
    print(f"Extraction      : {agent1['methode_extraction']}")
    print(f"Titre du cours  : {agent2.get('titre_cours')}")
    print(f"Source Agent 2  : {agent2.get('source')}")
    print(f"Chapitres       : {agent1['nb_chapitres']}")
    print(f"Unites de sens  : {agent3['nb_unites']} ({agent3['taille_moyenne_mots']} mots en moyenne)")

    _entete("Agents 4 et 5 — Vectorisation et recherche")
    print(f"Moteur          : {agent4['moteur']} ({agent4['dimension']} dimensions)")
    print(f"Type            : {agent4['type_moteur']}{'  [REPLI]' if agent4['repli_actif'] else ''}")
    print(f"Base vectorielle: {agent5['index']['moteur']}")
    print(f"Notions indexees: {agent5['nb_notions_indexees']}")

    _entete("Agent 6 — Comparaison")
    for pays in agent6["par_pays"].values():
        print(
            f"  {pays['pays']:<22} couverture {pays['taux_couverture_pct']:>5} %  "
            f"({pays['nb_couvertes']} couvertes / {pays['nb_partielles']} partielles / "
            f"{pays['nb_manquantes']} manquantes)"
        )
    print(f"\nScore global    : {agent6['score_global_pct']} %")
    print(f"Notions manquantes : {agent6['nb_notions_manquantes']}")
    print(f"Analyse qualitative : {'disponible' if agent6['gemini']['disponible'] else 'INDISPONIBLE'}")

    _entete("Agent 7 — Evaluation")
    print(f"Note globale    : {agent7['note_globale']}/100  ({agent7['niveau_maturite']})")
    for contribution in agent7["contributions"]:
        print(f"  {contribution['libelle']:<52} {contribution['valeur']:.2f}  [{contribution['niveau']}]")

    _entete("Agent 8 — Recommandations priorisees")
    for reco in agent8["recommandations"]:
        print(f"  {reco['rang']}. [{reco['priorite']:<8}] {reco['titre']}")
        print(f"     notion : {reco['notion']}  |  referentiels : {', '.join(reco['referentiels'])}")

    _entete("Agent 9 — Rapport final")
    print(agent9.get("synthese_executive", ""))
    fiabilite = agent9.get("fiabilite") or {}
    print(f"\nConfiance       : {fiabilite.get('niveau_confiance_pct')} %")
    for alerte in fiabilite.get("alertes", []):
        print(f"  ! {alerte}")
    print(f"\nDuree totale    : {analyse['duree_analyse_s']} s")

    if args.sortie_json:
        nettoye = module_traitement_analyse._nettoyer_pour_persistance(analyse)
        with open(args.sortie_json, "w", encoding="utf-8") as fichier:
            json.dump(nettoye, fichier, ensure_ascii=False, indent=2, default=str)
        print(f"\nResultat complet exporte vers {args.sortie_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
