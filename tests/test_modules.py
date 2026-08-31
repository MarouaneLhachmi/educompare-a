"""
Tests des modules fonctionnels et des regles d'acces.
======================================================

L'accent est mis sur deux familles de comportements :

- **la validation d'un depot**, qui est la seule barriere entre un fichier
  quelconque et une chaine de traitement de plusieurs dizaines de secondes ;
- **le cloisonnement des analyses**, qui est une exigence de securite : un
  utilisateur ne doit jamais consulter l'analyse d'un autre. Cette regle est
  verifiee de bout en bout, a travers de vraies requetes HTTP, parce qu'une
  regression peut aussi bien venir d'un decorateur oublie sur une route que de
  la fonction d'habilitation elle-meme.
"""

import io

import pytest

from app.modules import module_auth_securite, module_depot_documents
from app.services import database


# ---------------------------------------------------------------------------
# Module Depot — validation
# ---------------------------------------------------------------------------

class _FichierFactice:
    """Reproduit l'interface de `werkzeug.FileStorage` utilisee par `valider`."""

    def __init__(self, nom: str, contenu: bytes):
        self.filename = nom
        self.stream = io.BytesIO(contenu)


def _pdf_factice(taille: int = 4096) -> bytes:
    return b"%PDF-1.4\n" + b"0" * (taille - 9)


class TestValidationDepot:

    def test_pdf_valide_accepte(self):
        module_depot_documents.valider(_FichierFactice("cours.pdf", _pdf_factice()))

    def test_aucun_fichier(self):
        with pytest.raises(module_depot_documents.DocumentInvalide):
            module_depot_documents.valider(None)

    def test_nom_de_fichier_vide(self):
        with pytest.raises(module_depot_documents.DocumentInvalide):
            module_depot_documents.valider(_FichierFactice("", _pdf_factice()))

    def test_extension_refusee(self):
        with pytest.raises(module_depot_documents.DocumentInvalide) as erreur:
            module_depot_documents.valider(_FichierFactice("cours.docx", _pdf_factice()))
        assert "docx" in str(erreur.value)

    def test_fichier_trop_petit(self):
        with pytest.raises(module_depot_documents.DocumentInvalide):
            module_depot_documents.valider(_FichierFactice("cours.pdf", b"%PDF-1.4"))

    def test_fichier_trop_volumineux(self):
        from app.config import Config

        enorme = _pdf_factice(Config.MAX_CONTENT_LENGTH + 1024)
        with pytest.raises(module_depot_documents.DocumentInvalide) as erreur:
            module_depot_documents.valider(_FichierFactice("cours.pdf", enorme))
        assert "taille" in str(erreur.value).lower()

    def test_fichier_renomme_en_pdf_rejete(self):
        """
        Un exécutable ou une archive renommee en .pdf ne doit pas atteindre le
        pipeline : la signature du fichier est verifiee, pas seulement son nom.
        """
        faux = b"PK\x03\x04" + b"0" * 4096
        with pytest.raises(module_depot_documents.DocumentInvalide) as erreur:
            module_depot_documents.valider(_FichierFactice("cours.pdf", faux))
        assert "PDF" in str(erreur.value)

    def test_le_flux_reste_lisible_apres_validation(self):
        """`valider` mesure la taille et lit l'entete : il doit rembobiner."""
        fichier = _FichierFactice("cours.pdf", _pdf_factice())
        module_depot_documents.valider(fichier)
        assert fichier.stream.read(4) == b"%PDF"


class TestPreExtraction:

    def test_document_lisible(self, cours_complet):
        apercu = module_depot_documents.pre_extraire(cours_complet)
        assert apercu["lisible"] is True
        assert apercu["nb_mots"] > module_depot_documents.MOTS_MINIMUM_TRIAGE

    def test_document_sans_couche_texte_signale_non_lisible(self):
        from conftest import chemin_corpus

        apercu = module_depot_documents.pre_extraire(chemin_corpus("scan_sans_texte.pdf"))
        assert apercu["lisible"] is False
        assert apercu["nb_mots"] == 0

    def test_le_triage_alerte_sur_un_document_scanne(self):
        """
        Le module doit prevenir *avant* de lancer une analyse qui echouera a
        l'Agent 1 : c'est tout l'interet du controle au depot.
        """
        from conftest import chemin_corpus

        diagnostic = module_depot_documents.analyser_contenu(
            chemin_corpus("scan_sans_texte.pdf")
        )
        assert diagnostic["alertes"]
        assert any(a["niveau"] == "erreur" for a in diagnostic["alertes"])


class TestUtilitairesDepot:

    def test_taille_lisible(self):
        assert "Ko" in module_depot_documents.taille_lisible(2048)
        assert "Mo" in module_depot_documents.taille_lisible(5 * 1024 * 1024)

    def test_identifiant_valide(self):
        assert module_depot_documents.identifiant_valide("a1b2c3d4e5") is True
        assert module_depot_documents.identifiant_valide("../../etc/passwd") is False
        assert module_depot_documents.identifiant_valide("") is False

    def test_nom_de_fichier_accentue_translittere(self):
        """
        Comportement documente : « mathématiques » doit devenir
        « mathematiques » et non « mathmatiques ».
        """
        nom = module_depot_documents._nom_sur("Cours de mathématiques.pdf")
        assert "mathematiques" in nom

    def test_nom_de_fichier_vide_remplace(self):
        assert module_depot_documents._nom_sur("...").endswith(".pdf")


# ---------------------------------------------------------------------------
# Cloisonnement des analyses
# ---------------------------------------------------------------------------

AUTRE_UTILISATEUR = {
    "id": "utilisateur02", "email": "autre@example.org", "nom": "Autre enseignant",
    "role": "utilisateur", "actif": True,
}
ADMINISTRATEUR = {
    "id": "admin01", "email": "admin@example.org", "nom": "Administrateur",
    "role": "administrateur", "actif": True,
}


def _analyse_de(utilisateur_id: str | None, analyse_id: str = "an01") -> dict:
    analyse = {
        "id": analyse_id,
        "utilisateur_id": utilisateur_id,
        "nom_fichier": "cours.pdf",
        "statut": "TERMINEE",
        "matiere": "Mathématiques",
        "niveau": "Dernière année du primaire",
        "date_creation": "01/01/2026 10:00",
        "date_creation_iso": "2026-01-01T10:00:00",
    }
    database.creer_analyse(analyse)
    return analyse


class TestHabilitationUnitaire:

    def test_proprietaire_autorise(self, application, utilisateur_figee, creer_utilisateur):
        creer_utilisateur(utilisateur_figee)
        analyse = _analyse_de(utilisateur_figee["id"])
        with application.test_request_context():
            from flask import session

            session["utilisateur_id"] = utilisateur_figee["id"]
            assert module_auth_securite.peut_consulter_analyse(analyse) is True

    def test_tiers_refuse(self, application, utilisateur_figee, creer_utilisateur):
        creer_utilisateur(utilisateur_figee)
        creer_utilisateur(AUTRE_UTILISATEUR)
        analyse = _analyse_de(AUTRE_UTILISATEUR["id"])
        with application.test_request_context():
            from flask import session

            session["utilisateur_id"] = utilisateur_figee["id"]
            assert module_auth_securite.peut_consulter_analyse(analyse) is False

    def test_analyse_inexistante_refusee(self, application):
        with application.test_request_context():
            assert module_auth_securite.peut_consulter_analyse(None) is False

    def test_analyse_anonyme_consultable_par_le_porteur_du_lien(self, application):
        """
        Comportement volontaire du mode demonstration : une analyse sans
        proprietaire reste accessible. Le test fige ce choix pour qu'il soit
        modifie sciemment, et non par accident.
        """
        analyse = _analyse_de(None)
        with application.test_request_context():
            assert module_auth_securite.peut_consulter_analyse(analyse) is True


class TestCloisonnementHttp:
    """
    Meme regle, verifiee a travers les routes reellement exposees : une
    regression peut aussi bien venir d'un decorateur oublie que de la fonction
    d'habilitation. Le rendu complet d'un rapport demande une analyse reelle :
    il est verifie par `test_ancrage.py`, ici seuls les refus le sont.
    """

    def test_un_tiers_recoit_403(self, client, utilisateur_figee, connecter,
                                 creer_utilisateur):
        creer_utilisateur(AUTRE_UTILISATEUR)
        connecter(utilisateur_figee)
        _analyse_de(AUTRE_UTILISATEUR["id"])
        assert client.get("/rapport/an01").status_code == 403

    def test_analyse_inconnue_donne_404(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        assert client.get("/rapport/inexistante").status_code == 404

    def test_un_tiers_ne_peut_pas_supprimer(self, client, utilisateur_figee, connecter,
                                            creer_utilisateur):
        creer_utilisateur(AUTRE_UTILISATEUR)
        connecter(utilisateur_figee)
        _analyse_de(AUTRE_UTILISATEUR["id"])
        assert client.post("/espace/analyse/an01/supprimer").status_code == 403
        assert database.analyse_par_id("an01") is not None, (
            "l'analyse d'autrui a été supprimée malgré le refus"
        )

    def test_le_proprietaire_peut_supprimer(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"])
        assert client.post("/espace/analyse/an01/supprimer").status_code in (301, 302)
        assert database.analyse_par_id("an01") is None

    def test_le_back_office_est_ferme_aux_utilisateurs(self, client, utilisateur_figee,
                                                       connecter):
        connecter(utilisateur_figee)
        assert client.get("/administration/").status_code == 403

    def test_l_espace_utilisateur_exige_une_connexion(self, client):
        reponse = client.get("/espace/")
        assert reponse.status_code in (301, 302)
        assert "/connexion" in reponse.headers.get("Location", "")

    def test_l_historique_ne_montre_que_ses_propres_analyses(self, client, utilisateur_figee,
                                                             connecter, creer_utilisateur):
        creer_utilisateur(AUTRE_UTILISATEUR)
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"], "mienne")
        _analyse_de(AUTRE_UTILISATEUR["id"], "autrui")
        corps = client.get("/espace/historique").get_data(as_text=True)
        assert "mienne" in corps
        assert "autrui" not in corps


class TestAnnexeAccreditationHttp:
    """
    L'annexe suit exactement les memes regles d'acces que le rapport : elle en
    est un extrait, plus detaille sur la preuve. La generation elle-meme est
    verifiee par les tests d'ancrage, qui disposent d'une analyse reelle.
    """

    def test_un_tiers_recoit_403(self, client, utilisateur_figee, connecter,
                                 creer_utilisateur):
        creer_utilisateur(AUTRE_UTILISATEUR)
        connecter(utilisateur_figee)
        _analyse_de(AUTRE_UTILISATEUR["id"])
        assert client.get("/rapport/an01/annexe").status_code == 403

    def test_analyse_inconnue_donne_404(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        assert client.get("/rapport/inexistante/annexe").status_code == 404

    def test_analyse_non_terminee_redirige_vers_le_suivi(self, client,
                                                         utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        analyse = _analyse_de(utilisateur_figee["id"])
        database.mettre_a_jour_analyse(analyse["id"], {"statut": "EN_COURS"})
        reponse = client.get("/rapport/an01/annexe")
        assert reponse.status_code in (301, 302)
        assert "suivi" in reponse.headers.get("Location", "")


class TestRetourEnseignantHttp:
    """
    Route de la boucle de retour (plan de transition, phase 1.2, mode ombre) :
    l'accès suit exactement la même règle de cloisonnement que la lecture du
    rapport, puisqu'on ne peut pas commenter une notion qu'on n'a pas le
    droit de voir.
    """

    def _corps(self, **surcharge):
        corps = {"type": "couverture_notion", "cle_notion": "FR::Fractions",
                "valeur": "confirme"}
        corps.update(surcharge)
        return corps

    def test_le_proprietaire_peut_deposer_un_retour(self, client, utilisateur_figee,
                                                    connecter):
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"])
        reponse = client.post(f"/rapport/an01/retour", json=self._corps())
        assert reponse.status_code == 200
        assert reponse.get_json()["ok"] is True
        assert database.lister_retours("an01")[0]["utilisateur_id"] == utilisateur_figee["id"]

    def test_un_tiers_ne_peut_pas_deposer_de_retour(self, client, utilisateur_figee,
                                                    connecter, creer_utilisateur):
        creer_utilisateur(AUTRE_UTILISATEUR)
        connecter(utilisateur_figee)
        _analyse_de(AUTRE_UTILISATEUR["id"])
        reponse = client.post(f"/rapport/an01/retour", json=self._corps())
        assert reponse.status_code == 403
        assert database.lister_retours("an01") == []

    def test_analyse_inconnue(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        reponse = client.post("/rapport/inexistante/retour", json=self._corps())
        assert reponse.status_code == 404

    def test_type_invalide_rejete(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"])
        reponse = client.post("/rapport/an01/retour",
                              json=self._corps(type="type_qui_n_existe_pas"))
        assert reponse.status_code == 400
        assert database.lister_retours("an01") == []

    def test_cle_notion_manquante_rejetee(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"])
        reponse = client.post("/rapport/an01/retour", json=self._corps(cle_notion=""))
        assert reponse.status_code == 400

    def test_corps_absent_rejete(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        _analyse_de(utilisateur_figee["id"])
        reponse = client.post("/rapport/an01/retour")
        assert reponse.status_code == 400

    def test_le_retour_n_altere_pas_l_analyse(self, client, utilisateur_figee, connecter):
        """
        Propriété centrale du mode ombre : déposer une étiquette ne doit rien
        changer à l'analyse elle-même.
        """
        connecter(utilisateur_figee)
        avant = _analyse_de(utilisateur_figee["id"])
        client.post("/rapport/an01/retour", json=self._corps())
        apres = database.analyse_par_id("an01")
        assert apres["statut"] == avant["statut"]
        assert "retours" not in apres


class TestTrajectoireCloisonneeParVersion:
    """
    La trajectoire ecarte les analyses produites sur une autre version des
    referentiels. Ce filtrage est correct, mais il doit se voir : perdre des
    points de mesure en silence serait pire que ne pas filtrer du tout.
    """

    def _analyse_versionnee(self, identifiant, utilisateur_id, version, jour, note):
        database.creer_analyse({
            "id": identifiant,
            "utilisateur_id": utilisateur_id,
            "statut": "TERMINEE",
            "nom_fichier": "cours.pdf",
            "matiere": "Mathématiques",
            "niveau": "Dernière année du primaire",
            "resume_note_globale": note,
            "referentiel_version": version,
            "date_creation": f"{jour:02d}/01/2026 10:00",
            "date_creation_iso": f"2026-01-{jour:02d}T10:00:00",
        })

    def test_le_tableau_de_bord_signale_les_analyses_ecartees(
            self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        for jour in range(1, 5):
            self._analyse_versionnee(f"neuve{jour}", utilisateur_figee["id"],
                                     "FR:2.0-officiel", jour + 4, 60 + jour)
        self._analyse_versionnee("ancienne", utilisateur_figee["id"],
                                 "FR:1.0-reconstitue", 1, 40)

        corps = client.get("/espace/").get_data(as_text=True)
        assert "analyse(s) écartée(s)" in corps, (
            "l'analyse écartée de la trajectoire ne l'est pas visiblement"
        )
        assert "FR:2.0-officiel" in corps

    def test_aucune_mention_quand_tout_est_comparable(
            self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        for jour in range(1, 6):
            self._analyse_versionnee(f"a{jour}", utilisateur_figee["id"],
                                     "FR:1.0-reconstitue", jour, 50 + jour)

        corps = client.get("/espace/").get_data(as_text=True)
        assert "analyse(s) écartée(s)" not in corps


class TestPagesPubliques:

    def test_accueil_accessible_sans_compte(self, client):
        assert client.get("/").status_code == 200

    def test_page_de_connexion_accessible(self, client):
        assert client.get("/connexion").status_code == 200

    def test_api_referentiels(self, client):
        reponse = client.get("/api/referentiels")
        assert reponse.status_code == 200
        assert reponse.get_json()
