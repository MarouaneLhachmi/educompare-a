"""
Tests unitaires des services transverses.
==========================================

Ces services sont partages par plusieurs agents et plusieurs modules : une
regression y est donc particulierement couteuse, et particulierement discrete.
Chaque test cible une propriete que le code revendique explicitement dans ses
commentaires — c'est la seule facon de verifier une decision de conception
plutot qu'une implementation.
"""

import pytest

from app.services import (
    database,
    empreintes,
    gemini_client,
    pedagogie,
    profils_analyses,
    referentiels,
    synthese_extractive,
)


# ---------------------------------------------------------------------------
# database — repli en memoire compatible pymongo
# ---------------------------------------------------------------------------

class TestBaseEnMemoire:
    """Le repli memoire doit se comporter comme la fraction de pymongo utilisee."""

    def test_creation_et_relecture_d_une_analyse(self):
        database.creer_analyse({"id": "a1", "nom_fichier": "cours.pdf", "statut": "EN_COURS"})
        relue = database.analyse_par_id("a1")
        assert relue is not None
        assert relue["nom_fichier"] == "cours.pdf"

    def test_analyse_inconnue_retourne_none(self):
        assert database.analyse_par_id("inexistante") is None

    def test_enregistrer_analyse_fait_un_upsert(self):
        database.enregistrer_analyse({"id": "a2", "statut": "EN_COURS"})
        database.enregistrer_analyse({"id": "a2", "statut": "TERMINEE", "resume_note_globale": 61})
        assert database.compter_analyses() == 1
        assert database.analyse_par_id("a2")["statut"] == "TERMINEE"
        assert database.analyse_par_id("a2")["resume_note_globale"] == 61

    def test_lister_analyses_filtre_par_utilisateur(self):
        database.creer_analyse({"id": "a3", "utilisateur_id": "u1",
                                "date_creation_iso": "2026-01-01T10:00:00"})
        database.creer_analyse({"id": "a4", "utilisateur_id": "u2",
                                "date_creation_iso": "2026-01-02T10:00:00"})
        identifiants = {a["id"] for a in database.lister_analyses(utilisateur_id="u1")}
        assert identifiants == {"a3"}

    def test_lister_analyses_trie_du_plus_recent_au_plus_ancien(self):
        database.creer_analyse({"id": "ancienne", "date_creation_iso": "2026-01-01T10:00:00"})
        database.creer_analyse({"id": "recente", "date_creation_iso": "2026-06-01T10:00:00"})
        assert [a["id"] for a in database.lister_analyses()] == ["recente", "ancienne"]

    def test_suppression(self):
        database.creer_analyse({"id": "a5"})
        assert database.supprimer_analyse("a5") is True
        assert database.supprimer_analyse("a5") is False

    def test_operateurs_de_filtre_supportes(self):
        collection = database.get_db()["essai"]
        collection.insert_one({"id": "x", "code": "FR", "note": 10})
        collection.insert_one({"id": "y", "code": "UK", "note": 20})
        assert collection.count_documents({"code": {"$in": ["FR", "US"]}}) == 1
        assert collection.count_documents({"code": {"$ne": "FR"}}) == 1
        assert collection.count_documents({"note": {"$exists": True}}) == 2

    def test_journalisation_n_interrompt_jamais_le_parcours(self, monkeypatch):
        """
        La tracabilite est explicitement declaree non bloquante : une base
        indisponible ne doit pas faire echouer l'action de l'utilisateur.
        """
        def _casse():
            raise RuntimeError("base injoignable")

        monkeypatch.setattr(database, "get_db", _casse)
        database.journaliser("test", "u1", {})  # ne doit pas lever

    def test_isolation_entre_tests(self):
        """La fixture `base_memoire` doit repartir d'une base vide."""
        assert database.compter_analyses() == 0


class TestRetoursEnseignant:
    """
    Boucle de retour enseignant (plan de transition, phase 1.2), mode ombre :
    on collecte, rien d'autre ne change. Ces tests protegent la collecte
    elle-meme — la bascule sur ces donnees viendra plus tard.
    """

    def test_enregistrement_d_un_retour(self):
        database.enregistrer_retour(
            "a1", "couverture_notion", "FR::Fractions", "confirme", "u1"
        )
        retours = database.lister_retours("a1")
        assert len(retours) == 1
        assert retours[0]["type"] == "couverture_notion"
        assert retours[0]["cle_notion"] == "FR::Fractions"
        assert retours[0]["valeur"] == "confirme"

    def test_retour_anonyme_accepte(self):
        """Le mode démonstration sans compte doit pouvoir déposer un retour."""
        database.enregistrer_retour("a1", "qualite_exercice", "u001::exercice:0",
                                    "juste", utilisateur_id=None)
        assert database.lister_retours("a1")[0]["utilisateur_id"] is None

    def test_type_de_retour_inconnu_refuse(self):
        with pytest.raises(ValueError):
            database.enregistrer_retour("a1", "type_invalide", "FR::x", "oui", "u1")

    def test_cible_manquante_refusee(self):
        with pytest.raises(ValueError):
            database.enregistrer_retour("a1", "couverture_notion", "", "confirme", "u1")

    def test_lister_retours_filtre_par_analyse(self):
        database.enregistrer_retour("a1", "couverture_notion", "FR::x", "confirme", "u1")
        database.enregistrer_retour("a2", "couverture_notion", "FR::y", "infirme", "u1")
        assert len(database.lister_retours("a1")) == 1
        assert len(database.lister_retours()) == 2

    def test_compter_retours_par_type(self):
        database.enregistrer_retour("a1", "couverture_notion", "FR::x", "confirme", "u1")
        database.enregistrer_retour("a1", "couverture_notion", "FR::y", "infirme", "u1")
        database.enregistrer_retour("a1", "pertinence_recommandation", "FR::z", "utile", "u1")
        volume = database.compter_retours()
        assert volume["total"] == 3
        assert volume["par_type"]["couverture_notion"] == 2
        assert volume["par_type"]["pertinence_recommandation"] == 1
        assert volume["par_type"]["qualite_exercice"] == 0


# ---------------------------------------------------------------------------
# referentiels — base de connaissances
# ---------------------------------------------------------------------------

class TestReferentiels:

    def test_cles_disponibles_ecarte_les_metadonnees(self):
        cles = referentiels.cles_disponibles()
        assert cles
        assert all(not cle.startswith("_") for cle in cles)

    def test_toute_cle_est_un_couple_matiere_niveau(self):
        for cle in referentiels.cles_disponibles():
            assert " - " in cle, f"clé mal formée : {cle}"

    def test_cle_pour_retourne_la_cle_exacte(self):
        cle = referentiels.cles_disponibles()[0]
        matiere, _, niveau = cle.partition(" - ")
        assert referentiels.cle_pour(matiere, niveau) == cle

    def test_cle_pour_replie_sur_la_meme_matiere(self):
        """Niveau inconnu : le repli doit rester dans la bonne matiere."""
        cle = referentiels.cle_pour("Mathématiques", "Niveau qui n'existe pas")
        assert cle.startswith("Mathématiques - ")

    def test_cle_pour_ne_leve_jamais(self):
        assert referentiels.cle_pour("Matière inconnue", "Niveau inconnu") in \
            referentiels.cles_disponibles()

    def test_notions_a_plat_expose_le_texte_vectorise(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        notions = referentiels.notions_a_plat(cle)
        assert notions
        for notion in notions:
            assert set(notion) >= {"code", "pays", "notion", "descriptif", "texte"}
            # `texte` est la chaine reellement vectorisee par l'Agent 4 :
            # elle doit contenir l'intitule, sans quoi la recherche derive.
            assert notion["notion"] in notion["texte"]

    def test_notions_a_plat_filtre_par_pays(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        toutes = referentiels.notions_a_plat(cle)
        francaises = referentiels.notions_a_plat(cle, ["FR"])
        assert 0 < len(francaises) < len(toutes)
        assert {n["code"] for n in francaises} == {"FR"}

    def test_notions_a_plat_pays_inconnu_retourne_vide(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        assert referentiels.notions_a_plat(cle, ["ZZ"]) == []

    def test_statistiques_coherentes_avec_le_contenu(self):
        stats = referentiels.statistiques()
        total = sum(
            len(referentiels.notions_a_plat(cle))
            for cle in referentiels.cles_disponibles()
        )
        assert stats["nb_notions"] == total
        assert stats["nb_referentiels"] == len(referentiels.cles_disponibles())


class TestVersionnageReferentiels:
    """
    Versionnage des referentiels (plan de transition, phase 1.1).

    L'enjeu n'est pas technique mais probatoire : tant qu'une analyse ne dit
    pas sur quel socle de connaissances elle a ete produite, ses resultats ne
    sont ni reproductibles ni comparables entre eux.
    """

    def test_chaque_pays_declare_une_version_courante(self):
        codes = referentiels.codes_pays()
        assert codes
        for code in codes:
            assert referentiels.version_courante(code)

    def test_chaque_version_courante_est_publiee_au_manifeste(self):
        for code in referentiels.codes_pays():
            publiees = {v["version"] for v in referentiels.versions_disponibles(code)}
            assert referentiels.version_courante(code) in publiees, (
                f"{code} : la version courante n'est pas dans la liste publiée"
            )

    def test_charger_sans_version_resout_vers_la_courante(self):
        code = referentiels.codes_pays()[0]
        implicite = referentiels.charger(code)
        explicite = referentiels.charger(code, referentiels.version_courante(code))
        assert implicite == explicite

    def test_version_inconnue_leve(self):
        code = referentiels.codes_pays()[0]
        with pytest.raises(FileNotFoundError):
            referentiels.charger(code, "version-qui-n-existe-pas")

    def test_chaque_version_declare_sa_nature(self):
        """
        Un référentiel reconstitué ne doit jamais pouvoir passer pour le
        texte officiel : la nature est déclarée, pas déduite.
        """
        for code in referentiels.codes_pays():
            nature = referentiels.charger(code)["_meta"]["nature"]
            assert nature in {referentiels.NATURE_RECONSTITUE,
                             referentiels.NATURE_OFFICIEL}

    def test_un_referentiel_reconstitue_porte_son_avertissement(self):
        for code in referentiels.codes_pays():
            meta = referentiels.charger(code)["_meta"]
            if meta["nature"] == referentiels.NATURE_RECONSTITUE:
                assert meta.get("avertissement"), (
                    f"{code} : version reconstituée sans avertissement explicite"
                )

    def test_signature_couvre_tous_les_pays_du_referentiel(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        signature = referentiels.signature_versions(cle)
        attendus = {p["code"] for p in referentiels.pays_du_referentiel(cle)}
        assert set(signature["par_pays"]) == attendus
        assert signature["signature"]

    def test_signature_restreinte_aux_pays_demandes(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        signature = referentiels.signature_versions(cle, ["FR"])
        assert set(signature["par_pays"]) == {"FR"}
        assert signature["signature"].startswith("FR:")

    def test_signature_deterministe(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        assert (referentiels.signature_versions(cle)["signature"]
                == referentiels.signature_versions(cle)["signature"])

    def test_signature_signale_un_socle_non_officiel(self):
        """État actuel du projet : aucun référentiel n'est encore officiel."""
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        signature = referentiels.signature_versions(cle)
        assert signature["entierement_officiel"] is False
        assert referentiels.NATURE_RECONSTITUE in signature["natures"]

    def test_les_notions_portent_leur_version(self):
        cle = referentiels.cle_pour("Mathématiques", "Dernière année du primaire")
        for notion in referentiels.notions_a_plat(cle):
            assert notion["version"]

    def test_statistiques_exposent_la_provenance_par_pays(self):
        stats = referentiels.statistiques()
        assert len(stats["sources"]) == len(referentiels.codes_pays())
        assert stats["nb_officiels"] + stats["nb_reconstitues"] == len(stats["sources"])
        for source in stats["sources"]:
            assert source["version"]
            assert source["nature"]


class TestComparabiliteTrajectoire:
    """
    Une note ne veut rien dire hors du socle de connaissances qui l'a
    produite. La trajectoire doit donc se restreindre aux analyses partageant
    la version de la plus recente — sans quoi une revision du referentiel se
    lirait comme une evolution du travail de l'enseignant.
    """

    @staticmethod
    def _analyse(identifiant: str, note: float, version: str, jour: int) -> dict:
        return {
            "id": identifiant,
            "statut": "TERMINEE",
            "resume_note_globale": note,
            "referentiel_version": version,
            "date_creation": f"{jour:02d}/01/2026 10:00",
            "date_creation_iso": f"2026-01-{jour:02d}T10:00:00",
        }

    def test_les_analyses_d_une_autre_version_sont_ecartees(self):
        analyses = [
            self._analyse("v1a", 40, "FR:1.0", 1),
            self._analyse("v1b", 42, "FR:1.0", 2),
            self._analyse("v2a", 70, "FR:2.0", 3),
            self._analyse("v2b", 72, "FR:2.0", 4),
            self._analyse("v2c", 74, "FR:2.0", 5),
            self._analyse("v2d", 76, "FR:2.0", 6),
        ]
        trajectoire = profils_analyses.trajectoire(analyses)
        assert trajectoire["version_referentiel"] == "FR:2.0"
        assert trajectoire["nb_ecartees_autre_version"] == 2
        assert trajectoire["nb_points"] == 4
        # Sans le filtrage, le saut 42 -> 70 ferait apparaitre une pente
        # spectaculaire qui ne serait qu'un changement de referentiel.
        assert trajectoire["pente_par_analyse"] < 5

    def test_une_seule_version_ne_perd_aucune_analyse(self):
        analyses = [self._analyse(f"a{i}", 50 + i, "FR:1.0", i + 1) for i in range(5)]
        trajectoire = profils_analyses.trajectoire(analyses)
        assert trajectoire["nb_ecartees_autre_version"] == 0
        assert trajectoire["nb_points"] == 5

    def test_trop_peu_de_points_comparables_desactive_la_tendance(self):
        """
        Cinq analyses au total, mais une seule sur la version courante : il
        n'y a pas de tendance à lire, et le dire vaut mieux que l'inventer.
        """
        analyses = [self._analyse(f"v1{i}", 50, "FR:1.0", i + 1) for i in range(4)]
        analyses.append(self._analyse("v2a", 80, "FR:2.0", 9))
        trajectoire = profils_analyses.trajectoire(analyses)
        assert trajectoire["applique"] is False
        assert trajectoire["nb_ecartees_autre_version"] == 4

    def test_analyses_sans_version_restent_comparables_entre_elles(self):
        """Les analyses antérieures au versionnage ne doivent pas disparaître."""
        analyses = [
            {"id": f"a{i}", "statut": "TERMINEE", "resume_note_globale": 50 + i,
             "date_creation_iso": f"2026-01-0{i + 1}T10:00:00"}
            for i in range(4)
        ]
        trajectoire = profils_analyses.trajectoire(analyses)
        assert trajectoire["applique"] is True
        assert trajectoire["nb_points"] == 4


# ---------------------------------------------------------------------------
# gemini_client — extraction tolerante du JSON
# ---------------------------------------------------------------------------

class TestExtractionJson:
    """
    Le modele encadre regulierement sa reponse de balises markdown ou d'une
    phrase d'introduction. Chacun de ces cas a ete rencontre en production ;
    les perdre ferait basculer les agents generatifs en repli sans raison.
    """

    def test_json_nu(self):
        assert gemini_client.extract_json('{"note": 12}') == {"note": 12}

    def test_json_entoure_de_balises_markdown(self):
        assert gemini_client.extract_json('```json\n{"note": 12}\n```') == {"note": 12}

    def test_json_entoure_de_balises_sans_langage(self):
        assert gemini_client.extract_json('```\n{"note": 12}\n```') == {"note": 12}

    def test_json_precede_d_une_phrase(self):
        texte = 'Voici le résultat demandé :\n{"note": 12, "avis": "correct"}'
        assert gemini_client.extract_json(texte)["note"] == 12

    def test_tableau_json(self):
        assert gemini_client.extract_json('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]

    def test_reponse_sans_json_leve_une_erreur_explicite(self):
        with pytest.raises(ValueError):
            gemini_client.extract_json("Je ne peux pas répondre à cette demande.")

    def test_llm_neutralise_par_defaut(self):
        """Garde-fou : aucun test ne doit pouvoir appeler l'API reelle."""
        assert gemini_client.is_configured() is False
        with pytest.raises(gemini_client.LLMUnavailable):
            gemini_client.generate_text("test")


class TestLlmSimule:

    def test_le_client_simule_rend_les_reponses_deposees(self, llm_simule):
        llm_simule.repondre({"titre_cours": "Cours simulé"})
        assert gemini_client.generate_json("invite")["titre_cours"] == "Cours simulé"
        assert llm_simule.invites == ["invite"]

    def test_a_court_de_reponses_le_client_simule_devient_indisponible(self, llm_simule):
        with pytest.raises(gemini_client.LLMUnavailable):
            gemini_client.generate_text("invite")


# ---------------------------------------------------------------------------
# empreintes — MinHash / LSH
# ---------------------------------------------------------------------------

class TestEmpreintes:

    TEXTE = (
        "L'eleve apprend a comparer des fractions de meme denominateur et a calculer "
        "un pourcentage d'une quantite. Des exercices d'application sont proposes."
    )
    QUASI_DOUBLON = (
        "Les eleves apprennent a comparer des fractions de meme denominateur et a calculer "
        "un pourcentage d'une quantite. Des exercices d'application sont proposes."
    )
    AUTRE = (
        "Le circuit electrique comporte une pile, un interrupteur et une lampe. "
        "L'eleve distingue les materiaux conducteurs des materiaux isolants."
    )

    def test_empreinte_deterministe(self):
        assert empreintes.empreinte(self.TEXTE) == empreintes.empreinte(self.TEXTE)

    def test_similarite_a_soi_meme_vaut_un(self):
        signature = empreintes.empreinte(self.TEXTE)
        assert empreintes.similarite(signature, signature) == 1.0

    def test_quasi_doublon_detecte(self):
        score = empreintes.similarite(
            empreintes.empreinte(self.TEXTE), empreintes.empreinte(self.QUASI_DOUBLON)
        )
        assert score >= 0.55, f"quasi-doublon manqué (similarité {score})"

    def test_documents_differents_non_confondus(self):
        score = empreintes.similarite(
            empreintes.empreinte(self.TEXTE), empreintes.empreinte(self.AUTRE)
        )
        assert score < 0.2, f"faux positif de doublon (similarité {score})"

    def test_le_filtrage_lsh_ne_perd_pas_les_vrais_doublons(self):
        """
        Le LSH n'est qu'une optimisation : il doit ecarter des candidats, jamais
        changer le verdict sur un vrai doublon.
        """
        reference = empreintes.empreinte(self.TEXTE)
        candidats = [
            {"analyse_id": "doublon", "empreinte": empreintes.empreinte(self.QUASI_DOUBLON),
             "seaux": empreintes.seaux_lsh(empreintes.empreinte(self.QUASI_DOUBLON))},
            {"analyse_id": "different", "empreinte": empreintes.empreinte(self.AUTRE),
             "seaux": empreintes.seaux_lsh(empreintes.empreinte(self.AUTRE))},
        ]
        proches = empreintes.chercher_proches(reference, candidats)
        assert [p["analyse_id"] for p in proches] == ["doublon"]

    def test_candidat_sans_empreinte_ignore(self):
        reference = empreintes.empreinte(self.TEXTE)
        assert empreintes.chercher_proches(reference, [{"analyse_id": "x"}]) == []


# ---------------------------------------------------------------------------
# pedagogie — taxonomie de Bloom et lisibilite
# ---------------------------------------------------------------------------

class TestBloom:

    def test_verbe_de_bas_niveau(self):
        resultat = pedagogie.classer_bloom("Citer les unites de longueur du systeme metrique.")
        assert resultat["niveau"] in {"Mémoriser", "Comprendre"}
        assert resultat["indice"] <= 1

    def test_le_niveau_le_plus_eleve_l_emporte(self):
        """
        Choix de conception assume : un enonce qui demande de calculer *puis*
        de justifier releve du niveau superieur.
        """
        bas = pedagogie.classer_bloom("Calculer le perimetre du rectangle.")
        haut = pedagogie.classer_bloom(
            "Calculer le perimetre du rectangle puis justifier la demarche retenue."
        )
        assert haut["indice"] >= bas["indice"]

    def test_le_nom_formule_ne_declenche_pas_le_verbe_formuler(self):
        """
        Non-regression sur un faux positif documente dans `pedagogie.py` :
        un appariement par prefixe ferait basculer un simple calcul au niveau
        « Créer » a cause du mot « formule », tres frequent en mathematiques.
        """
        resultat = pedagogie.classer_bloom(
            "Appliquer la formule de l'aire du rectangle pour calculer la surface."
        )
        assert resultat["niveau"] != "Créer", (
            "« la formule » a été confondu avec le verbe « formuler »"
        )

    def test_profil_bloom_repartit_sur_cent(self):
        profil = pedagogie.profil_bloom([
            "Citer les unites de longueur.",
            "Calculer le perimetre d'un carre.",
            "Justifier le choix de la demarche.",
        ])
        assert abs(sum(profil["distribution_pct"].values()) - 100) < 1.5
        assert sum(profil["distribution"].values()) == 3
        assert 0.0 <= profil["profondeur_cognitive"] <= 1.0

    def test_profil_bloom_sur_liste_vide_ne_leve_pas(self):
        profil = pedagogie.profil_bloom([])
        assert profil["source"] == "aucun_enonce"
        assert profil["profondeur_cognitive"] == 0.0

    def test_la_profondeur_cognitive_croit_avec_le_niveau_des_enonces(self):
        superficiel = pedagogie.profil_bloom([
            "Citer les unites de longueur.", "Nommer les figures planes usuelles.",
        ])
        profond = pedagogie.profil_bloom([
            "Analyser la demarche proposee et justifier son choix.",
            "Concevoir un probleme original mobilisant les fractions.",
        ])
        assert profond["profondeur_cognitive"] > superficiel["profondeur_cognitive"]

    def test_lisibilite_croit_avec_la_simplicite(self):
        simple = pedagogie.indice_lisibilite(
            "Le chat dort. Le chien court. La balle roule. Le jour est beau."
        )
        complexe = pedagogie.indice_lisibilite(
            "L'interpretation phenomenologique des configurations epistemologiques "
            "contemporaines necessite une deconstruction methodologique prealable "
            "des presupposes conceptuels institutionnalises."
        )
        assert simple > complexe


# ---------------------------------------------------------------------------
# synthese_extractive — TextRank + MMR
# ---------------------------------------------------------------------------

class TestSyntheseExtractive:
    """
    L'objet de ces tests est l'algorithme — TextRank puis MMR — et non
    l'encodeur qui alimente la matrice de similarite. On force donc le repli
    lexical, documente dans le module : les proprietes verifiees (compression,
    fidelite a la source, ordre de lecture) sont les memes, et la suite reste
    instantanee au lieu de payer le chargement du modele semantique. Le
    chemin neuronal est exerce par les tests d'ancrage.
    """

    @pytest.fixture(autouse=True)
    def _repli_lexical(self, monkeypatch):
        from app.services import entrainement

        monkeypatch.setattr(entrainement, "encoder_textes",
                            lambda textes: (None, "neutralise (tests)"))

    TEXTE = (
        "Les fractions permettent de representer une partie d'un tout. "
        "L'eleve apprend a comparer deux fractions de meme denominateur. "
        "La simplification d'une fraction repose sur les diviseurs communs. "
        "Le perimetre d'un carre se calcule en multipliant le cote par quatre. "
        "L'aire du rectangle est le produit de la longueur par la largeur. "
        "Les unites de longueur se convertissent a l'aide d'un tableau. "
        "La masse se mesure en grammes et en kilogrammes. "
        "Un diagramme en batons represente des donnees chiffrees. "
        "La moyenne se calcule en divisant la somme par l'effectif. "
        "La proportionnalite se reconnait a un coefficient constant."
    )

    def test_la_synthese_compresse_le_texte(self):
        resultat = synthese_extractive.resumer(self.TEXTE, nb_phrases=4)
        assert resultat["disponible"] is True
        assert resultat["nb_phrases_retenues"] == 4
        assert resultat["taux_compression"] > 0.5

    def test_les_phrases_retenues_viennent_du_texte_source(self):
        """
        C'est une synthese *extractive* : aucune phrase ne doit etre inventee.
        La propriete est ce qui autorise a l'afficher sans relecture.
        """
        resultat = synthese_extractive.resumer(self.TEXTE, nb_phrases=4)
        for phrase in resultat["phrases"]:
            assert phrase["phrase"] in self.TEXTE

    def test_les_phrases_conservent_l_ordre_du_document(self):
        resultat = synthese_extractive.resumer(self.TEXTE, nb_phrases=4)
        positions = [p["position_source"] for p in resultat["phrases"]]
        assert positions == sorted(positions)

    def test_texte_trop_court_signale_sans_lever(self):
        resultat = synthese_extractive.resumer("Une seule phrase.", nb_phrases=4)
        assert resultat["disponible"] is False
        assert resultat["motif"]

    def test_le_repli_lexical_est_signale(self):
        """Le rapport affiche la provenance de la similarité : elle doit être dite."""
        resultat = synthese_extractive.resumer(self.TEXTE, nb_phrases=4)
        assert resultat["similarite"]


@pytest.mark.lent
class TestSyntheseAvecEncodeurReel:
    """
    Les memes proprietes, mais avec l'encodeur semantique reel : c'est le
    chemin nominal en production. Isole dans sa propre classe pour echapper au
    repli force ci-dessus, et marque `lent` a cause du chargement du modele.
    """

    def test_la_synthese_reste_extractive_et_ordonnee(self):
        resultat = synthese_extractive.resumer(TestSyntheseExtractive.TEXTE, nb_phrases=4)
        assert resultat["disponible"] is True
        assert resultat["nb_phrases_retenues"] == 4
        for phrase in resultat["phrases"]:
            assert phrase["phrase"] in TestSyntheseExtractive.TEXTE
        positions = [p["position_source"] for p in resultat["phrases"]]
        assert positions == sorted(positions)
