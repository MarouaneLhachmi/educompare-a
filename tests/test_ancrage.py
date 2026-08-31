"""
Tests d'ancrage — non-regression de bout en bout.
==================================================

Ces tests executent la chaine complete des neuf agents sur le corpus de
reference et verifient que les resultats restent dans les intervalles mesures
(`ancrages.json`). Ils sont marques `lent` et exclus de la commande par
defaut :

    pytest              # suite rapide (quelques secondes)
    pytest -m lent      # ancrages, chaine complete

Ce qu'ils protegent exactement
-------------------------------
Un changement de seuil dans l'Agent 6 ne casse rien, ne leve aucune exception
et ne se voit sur aucun ecran : il deplace silencieusement toutes les notes.
C'est arrive pendant le developpement — la note du cours de demonstration est
passee de 74,9 a 54. C'etait le resultat voulu (un diagnostic plus honnete),
mais rien dans le systeme ne l'aurait signale dans le cas contraire.

Un ancrage qui echoue ne signifie donc pas « le code est casse » : il signifie
« un resultat a bouge, justifie-le ». Si le deplacement est voulu, on relance
`python tests/mesurer_ancrages.py` et on commite la difference.

Deux familles de verifications cohabitent ici :

- les **intervalles absolus**, precis mais lies a l'environnement de mesure ;
- les **invariants d'ordre** (un cours complet couvre mieux qu'un plan de
  cours), plus grossiers mais vrais sur n'importe quelle machine. Ce sont eux
  qui portent le sens pedagogique du systeme.
"""

import os

import pytest

from app.modules import module_rapport_restitution, module_traitement_analyse
from conftest import charger_catalogue, chemin_corpus

pytestmark = [pytest.mark.lent, pytest.mark.ancrage]


# ---------------------------------------------------------------------------
# Execution du corpus, une seule fois pour toute la session
# ---------------------------------------------------------------------------

def _analyser(document: dict) -> dict:
    return module_traitement_analyse.run_analysis(
        pdf_path=chemin_corpus(document["fichier"]),
        matiere=document["matiere"],
        niveau=document["niveau"],
        nom_fichier_original=document["fichier"],
    )


@pytest.fixture(scope="session")
def analyses_du_corpus() -> dict[str, dict]:
    """
    Analyse tous les documents du corpus une fois pour toutes. Sans mise en
    cache, chaque test paierait de nouveau le cout complet du pipeline.

    Les fixtures `autouse` de `conftest.py` sont de portee fonction : elles ne
    couvrent pas l'execution d'une fixture de session. La neutralisation du
    modele de langage et de la base est donc reposee ici, dans les memes
    termes, pour que la mesure se fasse exactement dans les conditions des
    ancrages.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from app.services import database, gemini_client

    correctif = MonkeyPatch()

    def _indisponible(*args, **kwargs):
        raise gemini_client.LLMUnavailable("Modele de langage neutralise (ancrages).")

    correctif.setattr(gemini_client, "generate_text", _indisponible)
    correctif.setattr(gemini_client, "generate_json", _indisponible)
    correctif.setattr(gemini_client, "is_configured", lambda: False)
    correctif.setitem(database._STATE, "db", database.InMemoryDB())
    correctif.setitem(database._STATE, "mode", "memoire")

    resultats = {}
    for document in charger_catalogue():
        try:
            resultats[document["fichier"]] = _analyser(document)
        except Exception as exc:
            resultats[document["fichier"]] = {"statut": "ECHEC", "erreur": str(exc)}

    yield resultats
    correctif.undo()


def _documents(nature: str | None = None) -> list[dict]:
    return [d for d in charger_catalogue() if nature is None or d["nature"] == nature]


def _identifiants(documents: list[dict]) -> list[str]:
    return [d["fichier"] for d in documents]


# ---------------------------------------------------------------------------
# Intervalles mesures
# ---------------------------------------------------------------------------

class TestIntervallesAncres:

    @pytest.mark.parametrize(
        "document",
        [d for d in _documents() if d["attendu"]["extraction_reussie"]],
        ids=_identifiants([d for d in _documents() if d["attendu"]["extraction_reussie"]]),
    )
    def test_note_et_couverture_dans_l_intervalle(self, document, analyses_du_corpus,
                                                  ancrages):
        reference = (ancrages.get("documents") or {}).get(document["fichier"])
        if not reference or reference.get("statut") != "TERMINEE":
            pytest.skip(
                "Aucun ancrage mesuré pour ce document : lancer "
                "`python tests/mesurer_ancrages.py`."
            )

        analyse = analyses_du_corpus[document["fichier"]]
        assert analyse["statut"] == "TERMINEE", analyse.get("erreur")

        note = float((analyse["agent7"] or {}).get("note_globale") or 0)
        couverture = float((analyse["agent6"] or {}).get("score_global_pct") or 0)

        borne_basse, borne_haute = reference["intervalle"]["note_globale"]
        assert borne_basse <= note <= borne_haute, (
            f"note globale hors intervalle : {note} attendu dans "
            f"[{borne_basse} ; {borne_haute}] (mesure de référence : "
            f"{reference['mesure']['note_globale']}). Si le déplacement est "
            f"voulu, relancer `python tests/mesurer_ancrages.py`."
        )

        borne_basse, borne_haute = reference["intervalle"]["couverture_pct"]
        assert borne_basse <= couverture <= borne_haute, (
            f"couverture hors intervalle : {couverture} % attendu dans "
            f"[{borne_basse} ; {borne_haute}] (mesure de référence : "
            f"{reference['mesure']['couverture_pct']} %)."
        )

    def test_l_environnement_de_mesure_est_toujours_le_meme(self, analyses_du_corpus,
                                                            ancrages):
        """
        Un intervalle mesure avec le modele neuronal n'a pas de sens si la
        machine tourne desormais sur le repli LSA. Mieux vaut le dire que
        laisser interpreter l'ecart comme une regression du code.
        """
        reference = (ancrages.get("documents") or {}).get("cours_maths_complet.pdf")
        if not reference or reference.get("statut") != "TERMINEE":
            pytest.skip("Aucun ancrage mesuré.")

        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        attendu = reference["environnement"]
        assert bool((analyse["agent4"] or {}).get("repli_actif")) == \
            attendu["repli_vectorisation_actif"], (
                "le moteur de vectorisation n'est plus celui de la mesure : "
                "les intervalles ne sont pas comparables."
            )
        assert (analyse["agent6"] or {}).get("decision", {}).get("seuils") == \
            attendu["seuils"], (
                "les seuils de décision de l'Agent 6 ont changé — c'est "
                "précisément ce que ces tests sont là pour signaler."
            )


# ---------------------------------------------------------------------------
# Invariants pedagogiques
# ---------------------------------------------------------------------------

class TestInvariantsPedagogiques:
    """
    Ces proprietes ne dependent d'aucune valeur mesuree : elles disent ce que
    le systeme *doit* comprendre. Si l'une tombe, le diagnostic pedagogique
    est faux, quelle que soit la note affichee.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "DÉFAUT CONNU, révélé par ce corpus lors de la première mesure. "
            "`cours_maths_partiel.pdf` contient les 2 premiers chapitres de "
            "`cours_maths_complet.pdf` et rien d'autre, mais obtient 95,3 % de "
            "couverture contre 64,9 % pour le document complet : ajouter du "
            "contenu FAIT BAISSER la couverture mesurée. Cause : avec 2 unités "
            "seulement, le cross-encodeur note les paires disponibles très haut "
            "(logit +3,2 pour « Solides et volumes » face au chapitre sur les "
            "fractions, contre −2,3 face au chapitre de géométrie qui traite "
            "réellement les solides). Sur le document complet, le bi-encodeur "
            "présente d'autres candidats et masque l'effet. Sur les 385 paires "
            "du corpus, la corrélation de Spearman entre cosinus et logit du "
            "cross-encodeur n'est que de 0,11. La correction relève de la "
            "bascule vers `couverture_clf` entraîné sur des étiquettes réelles "
            "(phase 1.2), pas d'un ajustement de seuil. Retirer ce marqueur "
            "quand le défaut est corrigé — c'est ce que `strict=True` impose."
        ),
    )
    def test_un_cours_complet_couvre_mieux_qu_un_cours_partiel(self, analyses_du_corpus):
        """
        Propriete de monotonie : un sur-ensemble de contenu ne peut pas
        couvrir moins qu'un de ses sous-ensembles. C'est le minimum exigible
        d'une mesure de couverture.
        """
        complet = analyses_du_corpus["cours_maths_complet.pdf"]
        partiel = analyses_du_corpus["cours_maths_partiel.pdf"]
        assert complet["agent6"]["score_global_pct"] > partiel["agent6"]["score_global_pct"]

    def test_le_defaut_de_monotonie_reste_circonscrit(self, analyses_du_corpus):
        """
        Tant que le defaut ci-dessus n'est pas corrige, on surveille au moins
        qu'il ne s'aggrave pas : un document ampute de cinq chapitres sur sept
        ne doit pas depasser le document complet de plus de 35 points.
        """
        complet = analyses_du_corpus["cours_maths_complet.pdf"]["agent6"]["score_global_pct"]
        partiel = analyses_du_corpus["cours_maths_partiel.pdf"]["agent6"]["score_global_pct"]
        assert partiel - complet <= 35, (
            f"l'inversion de couverture s'aggrave : partiel {partiel} % contre "
            f"complet {complet} %"
        )

    def test_un_cours_complet_couvre_mieux_qu_un_plan_de_cours(self, analyses_du_corpus):
        """
        Le plan de cours enonce les memes intitules sans rien enseigner :
        c'est le cas limite qui justifie la separation entre probabilite de
        couverture et suffisance.
        """
        complet = analyses_du_corpus["cours_maths_complet.pdf"]
        plan = analyses_du_corpus["plan_de_cours.pdf"]
        assert complet["agent6"]["score_global_pct"] > plan["agent6"]["score_global_pct"]

    def test_un_cours_complet_est_mieux_note_qu_un_document_hors_sujet(self,
                                                                      analyses_du_corpus):
        complet = analyses_du_corpus["cours_maths_complet.pdf"]
        for hors_sujet in ("hors_sujet_cv.pdf", "hors_sujet_facture.pdf",
                           "hors_sujet_contrat.pdf"):
            assert complet["agent7"]["note_globale"] > \
                analyses_du_corpus[hors_sujet]["agent7"]["note_globale"], (
                    f"{hors_sujet} n'est pas distingué d'un vrai cours"
                )

    def test_un_quasi_doublon_donne_une_analyse_tres_proche(self, analyses_du_corpus):
        """
        Deux documents differant de quelques mots doivent aboutir au meme
        diagnostic. Un ecart important revelerait une instabilite de la chaine
        de decision.
        """
        original = analyses_du_corpus["cours_maths_complet.pdf"]
        doublon = analyses_du_corpus["cours_maths_doublon.pdf"]
        ecart_note = abs(original["agent7"]["note_globale"] - doublon["agent7"]["note_globale"])
        ecart_couverture = abs(
            original["agent6"]["score_global_pct"] - doublon["agent6"]["score_global_pct"]
        )
        assert ecart_note <= 6, f"écart de note trop important : {ecart_note}"
        assert ecart_couverture <= 10, f"écart de couverture trop important : {ecart_couverture}"

    def test_le_document_scanne_echoue_explicitement(self, analyses_du_corpus):
        analyse = analyses_du_corpus["scan_sans_texte.pdf"]
        assert analyse["statut"] == "ECHEC"
        assert analyse.get("erreur")

    def test_les_notions_manquantes_sont_justifiees(self, analyses_du_corpus):
        """
        Chaque verdict doit etre accompagne de sa preuve : c'est ce qui rend
        le rapport opposable plutot que persuasif.
        """
        agent6 = analyses_du_corpus["cours_maths_anglais.pdf"]["agent6"]
        assert agent6["notions_manquantes"]
        for notion in agent6["notions_manquantes"][:10]:
            assert notion.get("notion")
            assert notion.get("type_ecart") in {
                "absente", "amorcee", "evoquee_non_enseignee", "superficielle", "traitee"
            }
            assert notion.get("probabilite_couverture") is not None

    def test_la_couverture_reste_dans_les_bornes(self, analyses_du_corpus):
        for nom, analyse in analyses_du_corpus.items():
            if analyse["statut"] != "TERMINEE":
                continue
            couverture = analyse["agent6"]["score_global_pct"]
            note = analyse["agent7"]["note_globale"]
            assert 0 <= couverture <= 100, f"{nom} : couverture aberrante ({couverture})"
            assert 0 <= note <= 100, f"{nom} : note aberrante ({note})"


# ---------------------------------------------------------------------------
# Integrite de la chaine en mode degrade
# ---------------------------------------------------------------------------

class TestChaineEnModeDegrade:
    """
    Sans modele de langage, l'analyse doit **aboutir quand meme** et le dire.
    C'est la promesse centrale du systeme : c'est donc la premiere chose a
    proteger.
    """

    def test_les_neuf_agents_produisent_un_resultat(self, analyses_du_corpus):
        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        assert analyse["statut"] == "TERMINEE"
        for numero in range(1, 10):
            assert analyse.get(f"agent{numero}"), f"agent{numero} sans résultat"

    def test_aucun_agent_critique_en_incident(self, analyses_du_corpus):
        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        critiques = module_traitement_analyse.AGENTS_CRITIQUES
        incidents = {i["agent"] for i in analyse.get("incidents", [])}
        assert not (incidents & critiques), f"agents critiques en échec : {incidents & critiques}"

    def test_les_replis_sont_signales(self, analyses_du_corpus):
        """
        Le rapport doit dire quels replis ont ete actives, avec un niveau de
        confiance : un resultat degrade silencieusement serait pire qu'un
        resultat absent.
        """
        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        fiabilite = (analyse["agent9"] or {}).get("fiabilite") or {}
        assert fiabilite, "l'Agent 9 ne restitue aucune section de fiabilité"
        assert fiabilite.get("niveau_confiance_pct") is not None
        assert analyse["agent2"]["source"] == "repli_deterministe"

    def test_le_parcours_d_amelioration_est_produit_sans_llm(self, analyses_du_corpus):
        """
        Le parcours vient des modeles, pas du modele de langage : il doit
        exister meme hors ligne. Seule sa redaction est deleguee.
        """
        agent8 = analyses_du_corpus["cours_maths_anglais.pdf"]["agent8"]
        parcours = agent8.get("parcours_algorithmique") or {}
        assert parcours.get("disponible") is True
        assert parcours.get("nb_etapes", 0) > 0
        assert agent8.get("recommandations")

    def test_la_duree_est_mesuree(self, analyses_du_corpus):
        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        assert analyse["duree_analyse_s"] > 0
        assert analyse["durees_agents"]

    def test_chaque_analyse_trace_son_socle_de_connaissances(self, analyses_du_corpus):
        """
        Un resultat sans la version du referentiel qui l'a produit n'est ni
        reproductible ni comparable. La trace doit exister sur toute analyse
        aboutie, et couvrir exactement les pays interroges.
        """
        from app.services import referentiels

        for nom, analyse in analyses_du_corpus.items():
            if analyse["statut"] != "TERMINEE":
                continue
            assert analyse.get("referentiel_version"), f"{nom} : socle non tracé"
            par_pays = analyse["referentiel_versions_par_pays"]
            attendus = {
                p["code"]
                for p in referentiels.pays_du_referentiel(analyse["referentiel_utilise"])
            }
            assert set(par_pays) == attendus, f"{nom} : pays manquants dans la signature"
            assert analyse["referentiel_officiel"] is False, (
                f"{nom} : un référentiel se déclare officiel alors qu'aucun ne l'est encore"
            )


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------

class TestRestitution:

    def test_le_contexte_de_rapport_se_construit(self, analyses_du_corpus):
        contexte = module_rapport_restitution.build_report_context(
            analyses_du_corpus["cours_maths_complet.pdf"]
        )
        assert contexte
        for cle in ("agent6", "agent7", "agent8", "agent9"):
            assert cle in contexte, f"contexte de rapport incomplet : {cle} manquant"

    def test_le_rapport_html_s_affiche(self, client, connecter, utilisateur_figee,
                                       analyses_du_corpus):
        """Rendu reel du gabarit : un `Undefined` Jinja casserait la page."""
        from app.services import database

        connecter(utilisateur_figee)
        analyse = dict(analyses_du_corpus["cours_maths_complet.pdf"])
        analyse["utilisateur_id"] = utilisateur_figee["id"]
        database.enregistrer_analyse(
            module_traitement_analyse._nettoyer_pour_persistance(analyse)
        )
        reponse = client.get(f"/rapport/{analyse['id']}")
        assert reponse.status_code == 200

    def test_l_annexe_d_accreditation_se_genere(self, analyses_du_corpus, tmp_path):
        destination = str(tmp_path / "annexe.pdf")
        module_rapport_restitution.export_annexe_accreditation(
            analyses_du_corpus["cours_maths_complet.pdf"], destination
        )
        assert os.path.exists(destination)
        with open(destination, "rb") as fichier:
            assert fichier.read(4) == b"%PDF"

    def test_l_annexe_declare_sa_provenance_et_ses_angles_morts(
            self, analyses_du_corpus, tmp_path):
        """
        L'annexe est destinee a etre opposable : elle doit dire d'ou viennent
        ses mesures, sur quelle version du socle, et surtout ce qu'elle n'a
        pas tranche. Une annexe qui n'affirmerait que ses certitudes serait
        persuasive, pas opposable.
        """
        import pypdf

        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        destination = str(tmp_path / "annexe.pdf")
        module_rapport_restitution.export_annexe_accreditation(analyse, destination)

        texte = "\n".join(
            page.extract_text() for page in pypdf.PdfReader(destination).pages
        )

        # Provenance : version du socle et nature declarees.
        assert "Socle de connaissances" in texte
        assert "1.0-reconstitue" in texte
        assert "RECONSTITUE" in texte, (
            "un socle reconstitue doit etre signale comme tel, sans quoi "
            "l'annexe laisserait croire a un releve officiel"
        )
        # Angles morts : la section existe et nomme la zone d'incertitude.
        assert "n'a pas tranche" in texte
        assert "zone d'incertitude" in texte
        # Provenance technique rejouable.
        assert "Provenance des mesures" in texte
        assert (analyse["agent4"]["moteur"] or "")[:20] in texte

    def test_l_annexe_ne_contient_aucun_texte_genere(self, analyses_du_corpus,
                                                     tmp_path):
        """
        Aucune phrase du modele de langage ne doit passer dans l'annexe : la
        synthese executive de l'Agent 9 appartient au rapport, pas ici.
        """
        import pypdf

        analyse = analyses_du_corpus["cours_maths_complet.pdf"]
        destination = str(tmp_path / "annexe.pdf")
        module_rapport_restitution.export_annexe_accreditation(analyse, destination)

        texte = "\n".join(
            page.extract_text() for page in pypdf.PdfReader(destination).pages
        )
        synthese = ((analyse.get("agent9") or {}).get("synthese_executive") or "").strip()
        if len(synthese) > 40:
            assert synthese[:40] not in texte

    def test_l_export_pdf_se_genere(self, analyses_du_corpus, tmp_path):
        destination = str(tmp_path / "rapport.pdf")
        module_rapport_restitution.export_pdf(
            analyses_du_corpus["cours_maths_complet.pdf"], destination
        )
        assert os.path.exists(destination)
        assert os.path.getsize(destination) > 20_000, "export PDF anormalement léger"
        with open(destination, "rb") as fichier:
            assert fichier.read(4) == b"%PDF"
