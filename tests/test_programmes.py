"""
Tests du mode agrege : du cours au cursus (phase 2.1 du plan de transition).
============================================================================

L'agregation ne relance aucun calcul : elle recombine des decisions deja
prises. Elle peut donc etre testee sur des structures d'Agent 6 fabriquees a
la main, sans executer le pipeline — d'ou une suite rapide, capable de couvrir
des cas que le corpus de reference ne produit pas naturellement.

Deux familles de proprietes :

- **l'agregation elle-meme** : le maximum notion par notion, et la
  monotonie qui en decoule (ajouter un document ne peut pas faire baisser la
  couverture du cursus) ;
- **le tri de comparabilite** : melanger deux matieres ou deux versions de
  referentiel fabriquerait une couverture flatteuse et fausse. Ce qui est
  ecarte doit l'etre pour le bon motif, et se voir.
"""

import pytest

from app.agents import agent6_comparaison
from app.modules import module_historique_dashboard
from app.services import database


# ---------------------------------------------------------------------------
# Fabrication d'analyses figees
# ---------------------------------------------------------------------------

def _notion(intitule: str, probabilite: float, suffisance: float = 0.8,
            code: str = "FR") -> dict:
    """Ligne de l'Agent 6, dans la forme qu'il persiste reellement."""
    statut = agent6_comparaison._statut(probabilite)
    return {
        "notion": intitule,
        "descriptif": f"Descriptif de {intitule}",
        "pays": {"FR": "France", "UK": "Royaume-Uni"}.get(code, code),
        "code": code,
        "score": round(probabilite, 3),
        "probabilite_couverture": probabilite,
        "suffisance": suffisance,
        "statut": statut,
        "type_ecart": "traitee" if statut == "Couverte" else "absente",
        "libelle_ecart": "Traitée" if statut == "Couverte" else "Absente du support",
        "incertaine": False,
        "chapitre_correspondant": f"Chapitre sur {intitule}",
        "page_correspondante": 1,
        "extrait_correspondant": f"Extrait traitant de {intitule}.",
        "caracteristiques": {"cos_max": probabilite, "cross_prob_max": probabilite},
    }


def _analyse(identifiant: str, notions: list[dict], *, matiere="Mathématiques",
             niveau="Dernière année du primaire", version="FR:1.0",
             enseignant="Enseignant A", jour=1, statut="TERMINEE") -> dict:
    manquantes = [n for n in notions if n["statut"] == "Non couverte"]
    return {
        "id": identifiant,
        "statut": statut,
        "nom_fichier": f"{identifiant}.pdf",
        "titre_cours": f"Document {identifiant}",
        "matiere": matiere,
        "niveau": niveau,
        "referentiel_version": version,
        "utilisateur_id": "u1",
        "utilisateur_nom": enseignant,
        "date_creation": f"{jour:02d}/01/2026 10:00",
        "date_creation_iso": f"2026-01-{jour:02d}T10:00:00",
        "resume_note_globale": 60,
        "resume_couverture_pct": 50,
        "agent6": {
            "par_pays": {
                "FR": {
                    "code": "FR", "pays": "France", "drapeau": "🇫🇷",
                    "referentiel": "Programme de test",
                    "notions": notions,
                    "nb_notions": len(notions),
                },
            },
            "nb_notions_manquantes": len(manquantes),
            "reranking": {"applique": True},
            "score_global_pct": 50.0,
        },
    }


# ---------------------------------------------------------------------------
# Agregation
# ---------------------------------------------------------------------------

class TestAgregation:

    def test_aucune_analyse(self):
        agregat = agent6_comparaison.agreger_programme([])
        assert agregat["disponible"] is False
        assert agregat["motif"]

    def test_une_notion_couverte_par_un_seul_document_couvre_le_cursus(self):
        """
        C'est la raison d'etre du mode agrege : une notion absente du chapitre
        A mais traitee dans le chapitre B n'est pas un ecart du cursus.
        """
        a = _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.10)])
        b = _analyse("b", [_notion("Fractions", 0.12), _notion("Géométrie", 0.91)])

        agregat = agent6_comparaison.agreger_programme([a, b])
        par_notion = {n["notion"]: n for n in agregat["notions"]}
        assert par_notion["Fractions"]["statut"] == "Couverte"
        assert par_notion["Géométrie"]["statut"] == "Couverte"
        assert agregat["nb_notions_manquantes"] == 0

    def test_la_probabilite_agregee_est_le_maximum(self):
        a = _analyse("a", [_notion("Fractions", 0.40)])
        b = _analyse("b", [_notion("Fractions", 0.88)])
        agregat = agent6_comparaison.agreger_programme([a, b])
        assert agregat["notions"][0]["probabilite_couverture"] == pytest.approx(0.88)

    def test_la_suffisance_agregee_est_le_maximum_et_non_la_somme(self):
        """
        Choix de conception assume : additionner supposerait que les documents
        se completent plutot qu'ils ne se repetent, ce que rien n'etablit. Le
        maximum ne fabrique pas de couverture.
        """
        a = _analyse("a", [_notion("Fractions", 0.90, suffisance=0.30)])
        b = _analyse("b", [_notion("Fractions", 0.90, suffisance=0.45)])
        agregat = agent6_comparaison.agreger_programme([a, b])
        assert agregat["notions"][0]["suffisance"] == pytest.approx(0.45)

    def test_une_notion_absente_partout_reste_un_ecart(self):
        a = _analyse("a", [_notion("Solides", 0.10)])
        b = _analyse("b", [_notion("Solides", 0.15)])
        agregat = agent6_comparaison.agreger_programme([a, b])
        assert agregat["nb_notions_manquantes"] == 1
        assert agregat["notions_manquantes"][0]["notion"] == "Solides"

    def test_ajouter_un_document_ne_fait_jamais_baisser_la_couverture(self):
        """
        Monotonie : l'agregation par maximum l'impose. Sans elle, un
        etablissement aurait interet a cacher des documents.
        """
        a = _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.20)])
        b = _analyse("b", [_notion("Fractions", 0.30), _notion("Géométrie", 0.15)])
        seul = agent6_comparaison.agreger_programme([a])
        ensemble = agent6_comparaison.agreger_programme([a, b])
        assert ensemble["score_global_pct"] >= seul["score_global_pct"]

    def test_la_preuve_vient_du_document_qui_couvre_le_mieux(self):
        a = _analyse("a", [_notion("Fractions", 0.30)], enseignant="Enseignant A")
        b = _analyse("b", [_notion("Fractions", 0.93)], enseignant="Enseignant B")
        notion = agent6_comparaison.agreger_programme([a, b])["notions"][0]
        assert notion["analyse_referente"] == "b"
        assert notion["enseignant_referent"] == "Enseignant B"

    def test_les_contributeurs_sont_classes_du_meilleur_au_moins_bon(self):
        a = _analyse("a", [_notion("Fractions", 0.30)])
        b = _analyse("b", [_notion("Fractions", 0.93)])
        contributeurs = agent6_comparaison.agreger_programme([a, b])["notions"][0]["contributeurs"]
        assert [c["analyse_id"] for c in contributeurs] == ["b", "a"]

    def test_notions_exclusives_a_un_document(self):
        """
        Ce qui disparaitrait du cursus si ce document en etait retire — la
        mesure utile pour discuter la repartition entre enseignants.
        """
        a = _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.95)])
        b = _analyse("b", [_notion("Fractions", 0.90), _notion("Géométrie", 0.10)])
        documents = {
            d["analyse_id"]: d
            for d in agent6_comparaison.agreger_programme([a, b])["couverture_par_document"]
        }
        assert documents["a"]["nb_notions_exclusives"] == 1
        assert documents["a"]["notions_exclusives"] == ["Géométrie"]
        assert documents["b"]["nb_notions_exclusives"] == 0

    def test_les_notions_de_plusieurs_referentiels_restent_separees(self):
        a = _analyse("a", [_notion("Fractions", 0.95, code="FR"),
                           _notion("Fractions", 0.20, code="UK")])
        agregat = agent6_comparaison.agreger_programme([a])
        assert set(agregat["par_pays"]) == {"FR", "UK"}
        assert agregat["par_pays"]["FR"]["nb_couvertes"] == 1
        assert agregat["par_pays"]["UK"]["nb_couvertes"] == 0

    def test_analyse_sans_agent6_ignoree(self):
        a = _analyse("a", [_notion("Fractions", 0.95)])
        agregat = agent6_comparaison.agreger_programme([a, {"id": "vide", "statut": "TERMINEE"}])
        assert agregat["nb_documents"] == 1


# ---------------------------------------------------------------------------
# Comparabilite
# ---------------------------------------------------------------------------

class TestComparabiliteProgramme:

    @staticmethod
    def _programme(**surcharge) -> dict:
        programme = {
            "id": "p1", "nom": "Maths 6e", "matiere": "Mathématiques",
            "niveau": "Dernière année du primaire", "utilisateur_id": "u1",
            "analyse_ids": [],
        }
        programme.update(surcharge)
        return programme

    def _vue(self, programme: dict, analyses: list[dict]) -> dict:
        for analyse in analyses:
            database.creer_analyse(analyse)
        programme = dict(programme, analyse_ids=[a["id"] for a in analyses])
        database.creer_programme(programme)
        return module_historique_dashboard.vue_programme(
            database.programme_par_id(programme["id"])
        )

    def test_la_matiere_de_reference_vient_du_programme(self):
        """
        Non-regression. Deduire la matiere de l'analyse la plus recente rendait
        le perimetre du cursus dependant de l'ordre des depots : rattacher un
        document de sciences a un programme de mathematiques ecartait les
        mathematiques — l'inverse exact du comportement attendu.
        """
        maths_a = _analyse("m1", [_notion("Fractions", 0.95)], jour=1)
        maths_b = _analyse("m2", [_notion("Géométrie", 0.90)], jour=2)
        sciences = _analyse("s1", [_notion("Matière", 0.90)],
                            matiere="Sciences", jour=9)

        vue = self._vue(self._programme(), [maths_a, maths_b, sciences])

        assert {a["id"] for a in vue["analyses"]} == {"m1", "m2"}
        assert vue["nb_ecartees"] == 1
        assert vue["analyses_ecartees"][0]["id"] == "s1"

    def test_une_autre_version_de_referentiel_est_ecartee(self):
        ancienne = _analyse("v1", [_notion("Fractions", 0.95)], version="FR:1.0", jour=1)
        recente = _analyse("v2", [_notion("Fractions", 0.95)], version="FR:2.0", jour=5)

        vue = self._vue(self._programme(), [ancienne, recente])

        assert [a["id"] for a in vue["analyses"]] == ["v2"]
        assert vue["analyses_ecartees"][0]["motif_exclusion"] == \
            module_historique_dashboard.MOTIFS_EXCLUSION["version"]

    def test_une_analyse_non_terminee_est_ecartee(self):
        terminee = _analyse("ok", [_notion("Fractions", 0.95)])
        en_cours = _analyse("ko", [], statut="EN_COURS")

        vue = self._vue(self._programme(), [terminee, en_cours])

        assert [a["id"] for a in vue["analyses"]] == ["ok"]
        assert vue["analyses_ecartees"][0]["motif_exclusion"] == \
            module_historique_dashboard.MOTIFS_EXCLUSION["non_terminee"]

    def test_une_analyse_supprimee_depuis_est_signalee(self):
        """Le rattachement survit a la suppression : il faut le dire."""
        vivante = _analyse("vivante", [_notion("Fractions", 0.95)])
        database.creer_analyse(vivante)
        programme = self._programme(analyse_ids=["vivante", "disparue"])
        database.creer_programme(programme)

        vue = module_historique_dashboard.vue_programme(database.programme_par_id("p1"))
        motifs = [a["motif_exclusion"] for a in vue["analyses_ecartees"]]
        assert module_historique_dashboard.MOTIFS_EXCLUSION["introuvable"] in motifs

    def test_les_ecarts_resorbes_sont_comptes(self):
        a = _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.10)])
        b = _analyse("b", [_notion("Fractions", 0.10), _notion("Géométrie", 0.95)])

        vue = self._vue(self._programme(), [a, b])

        # Chaque document manque une notion isolement ; aucune ne manque au cursus.
        assert vue["manquantes_cumulees_isolement"] == 2
        assert vue["agregat"]["nb_notions_manquantes"] == 0
        assert vue["ecarts_resorbes"] == 2

    def test_la_matrice_couvre_toutes_les_notions_et_tous_les_documents(self):
        a = _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.10)])
        b = _analyse("b", [_notion("Fractions", 0.10), _notion("Géométrie", 0.95)])

        vue = self._vue(self._programme(), [a, b])
        matrice = vue["matrice"]

        assert matrice["nb_lignes"] == 2
        assert len(matrice["documents"]) == 2
        for ligne in matrice["lignes"]:
            assert len(ligne["cellules"]) == 2

    def test_programme_vide(self):
        vue = self._vue(self._programme(), [])
        assert vue["agregat"]["disponible"] is False
        assert vue["nb_analyses"] == 0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

AUTRE = {"id": "u2", "email": "autre@example.org", "nom": "Autre",
         "role": "utilisateur", "actif": True}


class TestRoutesProgrammes:

    def test_creation_et_consultation(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        reponse = client.post("/programmes/creer", data={
            "nom": "Maths 6e", "matiere": "Mathématiques",
            "niveau": "Dernière année du primaire", "annee": "2025-2026",
        }, follow_redirects=True)
        assert reponse.status_code == 200
        programmes = database.lister_programmes(utilisateur_figee["id"])
        assert len(programmes) == 1
        assert programmes[0]["nom"] == "Maths 6e"

    def test_creation_sans_nom_refusee(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        client.post("/programmes/creer", data={"nom": "  "}, follow_redirects=True)
        assert database.lister_programmes(utilisateur_figee["id"]) == []

    def test_un_tiers_ne_consulte_pas_le_programme_d_autrui(
            self, client, utilisateur_figee, connecter, creer_utilisateur):
        creer_utilisateur(AUTRE)
        connecter(utilisateur_figee)
        database.creer_programme({"id": "p9", "nom": "Privé", "utilisateur_id": AUTRE["id"]})
        assert client.get("/programmes/p9").status_code == 403

    def test_programme_inconnu(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        assert client.get("/programmes/inexistant").status_code == 404

    def test_connexion_requise(self, client):
        reponse = client.get("/programmes/")
        assert reponse.status_code in (301, 302)
        assert "/connexion" in reponse.headers.get("Location", "")

    def test_rattacher_l_analyse_d_autrui_est_refuse(
            self, client, utilisateur_figee, connecter, creer_utilisateur):
        """
        Un programme ne doit pas devenir un contournement du cloisonnement des
        analyses : on ne rattache que ce qu'on a le droit de lire.
        """
        creer_utilisateur(AUTRE)
        connecter(utilisateur_figee)
        database.creer_programme({"id": "p1", "nom": "Mien",
                                  "utilisateur_id": utilisateur_figee["id"]})
        database.creer_analyse({"id": "secrete", "statut": "TERMINEE",
                                "utilisateur_id": AUTRE["id"]})

        reponse = client.post("/programmes/p1/rattacher", data={"analyse_id": "secrete"})
        assert reponse.status_code == 403
        assert database.programme_par_id("p1")["analyse_ids"] == []

    def test_rattacher_puis_detacher(self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        database.creer_programme({"id": "p1", "nom": "Mien",
                                  "utilisateur_id": utilisateur_figee["id"]})
        database.creer_analyse({"id": "a1", "statut": "TERMINEE",
                                "nom_fichier": "cours.pdf",
                                "utilisateur_id": utilisateur_figee["id"]})

        client.post("/programmes/p1/rattacher", data={"analyse_id": "a1"})
        assert database.programme_par_id("p1")["analyse_ids"] == ["a1"]

        client.post("/programmes/p1/detacher/a1")
        assert database.programme_par_id("p1")["analyse_ids"] == []
        # Detacher ne supprime jamais l'analyse elle-meme.
        assert database.analyse_par_id("a1") is not None

    def test_supprimer_le_programme_conserve_les_analyses(
            self, client, utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        database.creer_programme({"id": "p1", "nom": "Mien",
                                  "utilisateur_id": utilisateur_figee["id"],
                                  "analyse_ids": ["a1"]})
        database.creer_analyse({"id": "a1", "statut": "TERMINEE",
                                "utilisateur_id": utilisateur_figee["id"]})

        client.post("/programmes/p1/supprimer")
        assert database.programme_par_id("p1") is None
        assert database.analyse_par_id("a1") is not None

    def test_la_page_programme_s_affiche(self, client, utilisateur_figee, connecter):
        """Rendu reel du gabarit, matrice comprise."""
        connecter(utilisateur_figee)
        analyses = [
            _analyse("a", [_notion("Fractions", 0.95), _notion("Géométrie", 0.10)]),
            _analyse("b", [_notion("Fractions", 0.10), _notion("Géométrie", 0.95)]),
        ]
        for analyse in analyses:
            analyse["utilisateur_id"] = utilisateur_figee["id"]
            database.creer_analyse(analyse)
        database.creer_programme({
            "id": "p1", "nom": "Maths 6e", "matiere": "Mathématiques",
            "niveau": "Dernière année du primaire",
            "utilisateur_id": utilisateur_figee["id"],
            "analyse_ids": ["a", "b"],
        })

        corps = client.get("/programmes/p1").get_data(as_text=True)
        assert "Maths 6e" in corps
        assert "Matrice notions" in corps
        assert "Fractions" in corps
