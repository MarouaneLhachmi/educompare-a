"""
Tests unitaires des agents, sur entrees figees.
===============================================

Chaque agent est teste **seul**, a partir de structures reproduisant la sortie
de l'agent amont (fixtures `agent1_figee`, `agent2_figee`). L'interet est
double : un echec designe directement le coupable, et la suite reste rapide,
la chaine complete n'etant executee que par les tests d'ancrage.

Les agents 4 a 9 chargent le modele semantique et sont donc testes dans
`test_ancrage.py` ; seules leurs fonctions de decision pures — celles qui
transforment un score en verdict — sont verifiees ici, car ce sont elles qui
fixent le sens des resultats et qui bougent quand on durcit un seuil.
"""

import pytest

from app.agents import (
    agent1_extraction,
    agent2_comprehension,
    agent3_decoupage,
    agent6_comparaison,
    agent7_evaluation,
)
from conftest import chemin_corpus


# ---------------------------------------------------------------------------
# Agent 1 — Extraction
# ---------------------------------------------------------------------------

class TestAgent1Extraction:

    def test_extraction_d_un_cours_complet(self, cours_complet):
        resultat = agent1_extraction.process(cours_complet)
        assert resultat["nb_pages"] >= 1
        assert resultat["nb_mots"] > 300
        assert resultat["ocr_utilise"] is False
        assert resultat["texte_complet"].strip()

    def test_detection_des_chapitres(self, cours_complet):
        """
        Le corpus de reference contient sept chapitres explicitement titres :
        le decoupage aval en depend entierement.
        """
        resultat = agent1_extraction.process(cours_complet)
        assert resultat["nb_chapitres"] >= 6
        assert all(chapitre["titre"] for chapitre in resultat["chapitres"])
        assert all(chapitre["contenu"] for chapitre in resultat["chapitres"])

    def test_langue_detectee_francais(self, cours_complet):
        assert agent1_extraction.process(cours_complet)["langue_detectee"] == "fr"

    def test_langue_detectee_anglais(self):
        resultat = agent1_extraction.process(chemin_corpus("cours_maths_anglais.pdf"))
        assert resultat["langue_detectee"] == "en"

    def test_fichier_absent(self):
        with pytest.raises(FileNotFoundError):
            agent1_extraction.process("chemin/qui/n/existe/pas.pdf")

    def test_pdf_sans_couche_texte_echoue_avec_un_message_exploitable(self):
        """
        Un document scanne sans OCR installe doit produire un message qui dit
        quoi faire, et non une exception technique. Le mode degrade documente
        cette situation explicitement.
        """
        with pytest.raises(ValueError) as erreur:
            agent1_extraction.process(chemin_corpus("scan_sans_texte.pdf"))
        message = str(erreur.value).lower()
        assert "ocr" in message or "scann" in message

    def test_extraction_deterministe(self, cours_complet):
        """Deux extractions du meme fichier doivent donner le meme texte."""
        premier = agent1_extraction.process(cours_complet)
        second = agent1_extraction.process(cours_complet)
        assert premier["texte_complet"] == second["texte_complet"]
        assert premier["nb_chapitres"] == second["nb_chapitres"]

    def test_mots_cles_ordonnes_par_frequence(self, cours_complet):
        mots_cles = agent1_extraction.process(cours_complet)["mots_cles"]
        occurrences = [m["occurrences"] for m in mots_cles]
        assert occurrences == sorted(occurrences, reverse=True)


# ---------------------------------------------------------------------------
# Agent 2 — Comprehension
# ---------------------------------------------------------------------------

class TestAgent2Comprehension:

    def test_repli_actif_sans_modele_de_langage(self, agent1_figee):
        """
        Propriete centrale du systeme : aucun agent generatif n'est bloquant.
        Sans LLM, l'Agent 2 doit produire une structure complete et le
        signaler.
        """
        resultat = agent2_comprehension.process(
            agent1_figee, "Mathématiques", "Dernière année du primaire"
        )
        assert resultat["source"] == "repli_deterministe"
        assert resultat["motif_repli"]
        assert len(resultat["chapitres"]) == len(agent1_figee["chapitres"])
        assert resultat["notions_cles_globales"]

    def test_chaque_chapitre_du_repli_porte_au_moins_un_objectif(self, agent1_figee):
        resultat = agent2_comprehension.process(
            agent1_figee, "Mathématiques", "Dernière année du primaire"
        )
        for chapitre in resultat["chapitres"]:
            assert chapitre["objectifs_pedagogiques"], (
                f"chapitre sans objectif : {chapitre['titre']}"
            )

    def test_chemin_nominal_avec_modele_de_langage(self, agent1_figee, llm_simule):
        llm_simule.repondre({
            "titre_cours": "Mathématiques — cycle 3",
            "resume": "Support couvrant fractions et géométrie.",
            "chapitres": [
                {"titre": "Chapitre 1 : Les fractions",
                 "objectifs_pedagogiques": ["Comparer des fractions"],
                 "notions_cles": ["fraction"]},
            ],
            "notions_cles_globales": ["fraction", "périmètre"],
        })
        resultat = agent2_comprehension.process(
            agent1_figee, "Mathématiques", "Dernière année du primaire"
        )
        assert resultat["source"] == "gemini"
        assert resultat["titre_cours"] == "Mathématiques — cycle 3"

    def test_reponse_llm_hors_format_bascule_en_repli(self, agent1_figee, llm_simule):
        """Une reponse inexploitable ne doit pas faire echouer l'agent."""
        llm_simule.repondre("Je ne peux pas répondre à cette demande.")
        resultat = agent2_comprehension.process(
            agent1_figee, "Mathématiques", "Dernière année du primaire"
        )
        assert resultat["source"] == "repli_deterministe"

    def test_document_sans_chapitre(self, agent1_figee):
        agent1_figee["chapitres"] = []
        agent1_figee["nb_chapitres"] = 0
        resultat = agent2_comprehension.process(
            agent1_figee, "Mathématiques", "Dernière année du primaire"
        )
        assert resultat["chapitres"] == []
        assert resultat["titre_cours"]


# ---------------------------------------------------------------------------
# Agent 3 — Decoupage
# ---------------------------------------------------------------------------

class TestAgent3Decoupage:

    def test_production_d_unites(self, agent1_figee, agent2_figee):
        resultat = agent3_decoupage.process(agent1_figee, agent2_figee)
        assert resultat["nb_unites"] >= len(agent1_figee["chapitres"])
        assert resultat["nb_unites"] == len(resultat["unites"])
        assert resultat["taille_moyenne_mots"] > 0

    def test_identifiants_uniques_et_ordonnes(self, agent1_figee, agent2_figee):
        unites = agent3_decoupage.process(agent1_figee, agent2_figee)["unites"]
        identifiants = [u["id"] for u in unites]
        assert len(set(identifiants)) == len(identifiants)
        assert identifiants == sorted(identifiants)

    def test_le_titre_du_chapitre_prefixe_la_premiere_unite(self, agent1_figee, agent2_figee):
        """
        Choix documente dans l'Agent 3 : le titre porte une forte charge
        semantique et ameliore sensiblement la recherche de l'Agent 5. Le
        perdre degraderait la couverture sans erreur visible.
        """
        unites = agent3_decoupage.process(agent1_figee, agent2_figee)["unites"]
        premieres = [u for u in unites if u["position"] == 0]
        assert premieres
        for unite in premieres:
            assert unite["texte"].startswith(unite["chapitre"])

    def test_chaque_unite_est_rattachee_a_son_chapitre(self, agent1_figee, agent2_figee):
        unites = agent3_decoupage.process(agent1_figee, agent2_figee)["unites"]
        titres = {c["titre"] for c in agent1_figee["chapitres"]}
        assert {u["chapitre"] for u in unites} <= titres

    def test_document_sans_structure_produit_une_unite_de_secours(self, agent1_figee,
                                                                  agent2_figee):
        """
        Sans cette unite de secours, les agents 4 a 6 recevraient une liste
        vide et l'analyse s'arreterait sur un agent critique.
        """
        agent1_figee["chapitres"] = []
        resultat = agent3_decoupage.process(agent1_figee, agent2_figee)
        assert resultat["nb_unites"] == 1
        assert resultat["unites"][0]["texte"].strip()

    def test_aucune_unite_vide(self, agent1_figee, agent2_figee):
        unites = agent3_decoupage.process(agent1_figee, agent2_figee)["unites"]
        for unite in unites:
            assert unite["texte"].strip()
            assert unite["nb_mots"] > 0


# ---------------------------------------------------------------------------
# Agent 6 — regles de decision
# ---------------------------------------------------------------------------

class TestAgent6Decision:
    """
    Ces fonctions traduisent un score en verdict affiche a l'enseignant. Un
    deplacement de seuil y est invisible a l'execution mais deplace toutes les
    notes : c'est exactement le scenario que les tests d'ancrage ont pour but
    d'attraper, et ces tests-ci en donnent la cause immediate.
    """

    def test_statut_croissant_avec_la_probabilite(self):
        assert agent6_comparaison._statut(0.95) == "Couverte"
        assert agent6_comparaison._statut(0.50) == "Partiellement couverte"
        assert agent6_comparaison._statut(0.05) == "Non couverte"

    def test_seuils_ordonnes(self):
        assert 0 < agent6_comparaison.SEUIL_PARTIELLE < agent6_comparaison.SEUIL_COUVERTE < 1

    def test_statut_exactement_au_seuil(self):
        """Les seuils sont inclusifs : la frontiere doit rester stable."""
        assert agent6_comparaison._statut(agent6_comparaison.SEUIL_COUVERTE) == "Couverte"
        assert agent6_comparaison._statut(
            agent6_comparaison.SEUIL_PARTIELLE) == "Partiellement couverte"

    @staticmethod
    def _valeurs(cos_max: float, cross: float) -> dict:
        return {"cos_max": cos_max, "cross_prob_max": cross}

    def test_notion_traitee(self):
        ecart = agent6_comparaison._type_ecart(
            0.90, 0.80, self._valeurs(0.80, 0.85), cross_actif=True
        )
        assert ecart == "traitee"

    def test_notion_traitee_trop_brievement(self):
        """Couverte mais sans matiere suffisante pour etre apprise."""
        ecart = agent6_comparaison._type_ecart(
            0.90, 0.10, self._valeurs(0.80, 0.85), cross_actif=True
        )
        assert ecart == "superficielle"

    def test_notion_evoquee_mais_non_enseignee(self):
        """
        Le cas qui a motive l'ajout du cross-encodeur : le theme est present
        (cosinus eleve), l'enseignement effectif ne l'est pas.
        """
        ecart = agent6_comparaison._type_ecart(
            0.30, 0.20, self._valeurs(0.75, 0.10), cross_actif=True
        )
        assert ecart == "evoquee_non_enseignee"

    def test_notion_absente(self):
        ecart = agent6_comparaison._type_ecart(
            0.05, 0.05, self._valeurs(0.20, 0.05), cross_actif=True
        )
        assert ecart == "absente"

    def test_notion_amorcee(self):
        ecart = agent6_comparaison._type_ecart(
            0.45, 0.30, self._valeurs(0.30, 0.60), cross_actif=True
        )
        assert ecart == "amorcee"

    def test_tout_type_d_ecart_possede_un_libelle_affichable(self):
        valeurs = [
            agent6_comparaison._type_ecart(p, s, self._valeurs(c, x), cross_actif=actif)
            for p, s, c, x, actif in [
                (0.9, 0.8, 0.8, 0.9, True), (0.9, 0.1, 0.8, 0.9, True),
                (0.3, 0.2, 0.75, 0.1, True), (0.45, 0.3, 0.3, 0.6, True),
                (0.05, 0.05, 0.2, 0.05, True), (0.3, 0.2, 0.75, 0.1, False),
            ]
        ]
        for ecart in valeurs:
            assert ecart in agent6_comparaison.LIBELLES_ECART, f"écart sans libellé : {ecart}"


# ---------------------------------------------------------------------------
# Agent 7 — echelle de maturite
# ---------------------------------------------------------------------------

class TestAgent7Maturite:

    def test_bornes_de_l_echelle(self):
        assert agent7_evaluation._niveau_maturite(100)[0] == "Aligné"
        assert agent7_evaluation._niveau_maturite(0)[0] == "À refondre"

    def test_echelle_monotone(self):
        """Une note superieure ne doit jamais donner un niveau inferieur."""
        ordre = ["À refondre", "À renforcer", "À consolider", "Solide", "Aligné"]
        rangs = [ordre.index(agent7_evaluation._niveau_maturite(n)[0]) for n in range(0, 101, 5)]
        assert rangs == sorted(rangs)

    def test_chaque_niveau_est_accompagne_d_une_explication(self):
        for note in (10, 40, 55, 70, 90):
            niveau, explication = agent7_evaluation._niveau_maturite(note)
            assert niveau and len(explication) > 20
