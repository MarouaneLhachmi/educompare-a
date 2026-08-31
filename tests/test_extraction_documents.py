"""
Tests de la lecture multi-format (phase 2.2 du plan de transition).
====================================================================

Deux exigences se croisent ici.

**Lire ce que l'auteur a ecrit.** Un `.docx` met souvent ses exercices dans un
tableau, un `.pptx` son raisonnement pedagogique dans les notes du
presentateur. Ne lire que le corps de texte perdrait la partie la plus
pedagogique du document, sans que rien ne le signale.

**Ne pas inventer ce qu'on ne sait pas.** Un `.docx` n'a pas de pages : sa
pagination depend de la police et de l'imprimante. Fabriquer des numeros les
ferait apparaitre dans l'annexe d'accreditation comme des preuves localisees.
Le document est donc restitue en une seule page, et le declare.
"""

import pytest

from app.agents import agent1_extraction
from app.services import extraction_documents
from app.services.extraction_documents import DocumentIllisible
from conftest import chemin_corpus


# ---------------------------------------------------------------------------
# Detection de format et dependances
# ---------------------------------------------------------------------------

class TestFormats:

    def test_format_deduit_de_l_extension(self):
        assert extraction_documents.format_de("cours.PDF") == "pdf"
        assert extraction_documents.format_de("/a/b/cours.docx") == "docx"
        assert extraction_documents.format_de("sans_extension") == ""

    def test_formats_supportes(self):
        assert extraction_documents.format_supporte("cours.pdf") is True
        assert extraction_documents.format_supporte("cours.pptx") is True
        assert extraction_documents.format_supporte("cours.txt") is False

    def test_les_dependances_sont_decrites(self):
        """
        La supervision technique doit pouvoir dire ce qui manque. L'OCR compte
        une piece hors de Python — le binaire Tesseract — qu'un simple import
        ne suffit pas a detecter.
        """
        etat = extraction_documents.dependances()
        assert set(etat) == {"pdf", "docx", "pptx", "ocr"}
        for lecteur in etat.values():
            assert "disponible" in lecteur and "module" in lecteur
        assert {"python_installe", "binaire_installe"} <= set(etat["ocr"])

    def test_fichier_absent(self):
        with pytest.raises(FileNotFoundError):
            extraction_documents.lire("chemin/inexistant.pdf")

    def test_format_non_supporte_donne_un_message_utile(self):
        with pytest.raises(DocumentIllisible) as erreur:
            extraction_documents.lire("requirements.txt")
        message = str(erreur.value)
        assert ".txt" in message
        assert ".pdf" in message  # le message dit ce qui EST accepte


# ---------------------------------------------------------------------------
# Documents Word
# ---------------------------------------------------------------------------

class TestDocx:

    def test_lecture(self):
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.docx"))
        assert lecture["format"] == "docx"
        assert "\n".join(lecture["pages"]).strip()
        assert lecture["ocr_utilise"] is False

    def test_les_styles_de_titre_sont_restitues(self):
        """
        C'est l'apport propre du format : la structure est declaree, pas
        devinee. Les intitules du corpus ne contiennent pas le mot
        « Chapitre » — l'heuristique du PDF les manquerait.
        """
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.docx"))
        assert lecture["titres"]
        assert any("grands nombres" in t.lower() for t in lecture["titres"])

    def test_le_contenu_des_tableaux_est_lu(self):
        texte = "\n".join(extraction_documents.lire(chemin_corpus("cours_maths.docx"))["pages"])
        assert "Convertir 3,5 km" in texte, (
            "les tableaux portent souvent les exercices : les ignorer perdrait "
            "une part du contenu pédagogique"
        )

    def test_la_pagination_est_declaree_non_fiable(self):
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.docx"))
        assert lecture["pagination_fiable"] is False
        assert len(lecture["pages"]) == 1


# ---------------------------------------------------------------------------
# Presentations
# ---------------------------------------------------------------------------

class TestPptx:

    def test_lecture(self):
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.pptx"))
        assert lecture["format"] == "pptx"
        assert len(lecture["pages"]) > 1

    def test_une_page_par_diapositive(self):
        """Une diapositive est une vraie unite : sa numerotation a un sens."""
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.pptx"))
        assert lecture["pagination_fiable"] is True
        assert len(lecture["pages"]) == lecture["nb_pages_document"]

    def test_les_titres_de_diapositive_sont_restitues(self):
        lecture = extraction_documents.lire(chemin_corpus("cours_maths.pptx"))
        assert lecture["titres"]

    def test_les_notes_du_presentateur_sont_lues(self):
        texte = "\n".join(extraction_documents.lire(chemin_corpus("cours_maths.pptx"))["pages"])
        assert "Note du presentateur" in texte, (
            "les notes portent souvent le raisonnement pédagogique que la "
            "diapositive se contente de résumer"
        )

    def test_max_pages_limite_la_lecture(self):
        complet = extraction_documents.lire(chemin_corpus("cours_maths.pptx"))
        partiel = extraction_documents.lire(chemin_corpus("cours_maths.pptx"), max_pages=2)
        assert len(partiel["pages"]) == 2
        assert len(complet["pages"]) > 2
        # Le nombre de pages du document reste celui du document, pas celui lu.
        assert partiel["nb_pages_document"] == complet["nb_pages_document"]


# ---------------------------------------------------------------------------
# PDF et OCR
# ---------------------------------------------------------------------------

class TestPdfEtOcr:

    def test_pdf_avec_couche_texte(self):
        lecture = extraction_documents.lire(chemin_corpus("cours_maths_complet.pdf"))
        assert lecture["format"] == "pdf"
        assert lecture["ocr_utilise"] is False
        assert lecture["pagination_fiable"] is True

    def test_pdf_sans_couche_texte_oriente_vers_l_ocr(self):
        """
        Sans OCR installe, le message doit dire quoi faire. C'est le mode
        degrade documente, pas un plantage.
        """
        with pytest.raises(DocumentIllisible) as erreur:
            extraction_documents.lire(chemin_corpus("scan_sans_texte.pdf"))
        message = str(erreur.value).lower()
        assert "ocr" in message or "scann" in message

    def test_l_ocr_peut_etre_desactive(self):
        """`tenter_ocr=False` sert au controle au depot, qui doit rester instantane."""
        with pytest.raises(DocumentIllisible):
            extraction_documents.lire(chemin_corpus("scan_sans_texte.pdf"),
                                      tenter_ocr=False)


# ---------------------------------------------------------------------------
# Integration avec l'Agent 1
# ---------------------------------------------------------------------------

class TestAgent1MultiFormat:

    @pytest.mark.parametrize("fichier,format_attendu", [
        ("cours_maths_complet.pdf", "pdf"),
        ("cours_maths.docx", "docx"),
        ("cours_maths.pptx", "pptx"),
    ])
    def test_l_agent_1_lit_les_trois_formats(self, fichier, format_attendu):
        resultat = agent1_extraction.process(chemin_corpus(fichier))
        assert resultat["format_source"] == format_attendu
        assert resultat["nb_mots"] > 200
        assert resultat["nb_chapitres"] >= 3
        assert resultat["texte_complet"].strip()

    def test_la_structure_declaree_est_preferee_a_l_heuristique(self):
        """
        Les sections du corpus bureautique ne contiennent pas le mot
        « Chapitre » : sans les styles de titre, l'heuristique retomberait sur
        son decoupage mecanique en « Section 1, 2, 3… ».
        """
        resultat = agent1_extraction.process(chemin_corpus("cours_maths.docx"))
        assert resultat["structure_declaree"] is True
        titres = [c["titre"] for c in resultat["chapitres"]]
        assert any("grands nombres" in t.lower() for t in titres)
        assert not any(t.startswith("Section ") for t in titres)

    def test_le_pdf_reste_sur_l_heuristique(self):
        resultat = agent1_extraction.process(chemin_corpus("cours_maths_complet.pdf"))
        assert resultat["structure_declaree"] is False
        assert resultat["nb_chapitres"] >= 6

    def test_format_refuse_avec_un_message_affichable(self):
        """Le contrat de l'agent reste `ValueError` : le pipeline s'y appuie."""
        with pytest.raises(ValueError) as erreur:
            agent1_extraction.process("requirements.txt")
        assert "non pris en charge" in str(erreur.value)


# ---------------------------------------------------------------------------
# Controle au depot
# ---------------------------------------------------------------------------

class TestDepotMultiFormat:

    def test_pre_extraction_docx(self):
        from app.modules import module_depot_documents

        apercu = module_depot_documents.pre_extraire(chemin_corpus("cours_maths.docx"))
        assert apercu["lisible"] is True
        assert apercu["format"] == "docx"

    def test_pre_extraction_pptx(self):
        from app.modules import module_depot_documents

        apercu = module_depot_documents.pre_extraire(chemin_corpus("cours_maths.pptx"))
        assert apercu["lisible"] is True
        assert apercu["format"] == "pptx"

    def test_le_triage_propose_l_ocr_au_lieu_de_refuser(self):
        """
        Changement de posture voulu en phase 2.2 : le module signale ce qui est
        possible plutot que de renvoyer l'utilisateur a un echec.
        """
        from app.modules import module_depot_documents

        diagnostic = module_depot_documents.analyser_contenu(
            chemin_corpus("scan_sans_texte.pdf")
        )
        assert diagnostic["ocr"]["propose"] is True
        message = diagnostic["alertes"][0]["message"].lower()
        assert "reconnaissance optique" in message
        # Le niveau depend de la presence reelle de l'OCR sur la machine.
        attendu = "alerte" if diagnostic["ocr"]["disponible"] else "erreur"
        assert diagnostic["alertes"][0]["niveau"] == attendu
