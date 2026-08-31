"""
Tests de la couche de présentation : tri, pagination, rendu des gabarits.
=========================================================================

Deux familles.

**Tri et pagination** — ils ne changent rien à ce que le système mesure, mais
une erreur y est invisible : une page 2 qui perd le filtrage, ou un tri qui
range les analyses sans note au milieu, ne lève aucune exception.

**Rendu des gabarits** — chaque page publique est demandée et doit répondre
200. Une variable oubliée dans un `render_template` ne casse rien avant qu'un
utilisateur n'ouvre la page ; ce filet l'attrape au premier `pytest`. Il a été
écrit après avoir laissé passer exactement ce cas : un gabarit mis à jour, la
route qui ne lui fournissait pas encore sa variable, et une erreur 500
découverte à la main dans le navigateur.
"""

import pytest

from app.routes import _presentation
from app.services import database


# ---------------------------------------------------------------------------
# Tri
# ---------------------------------------------------------------------------

def _analyse(identifiant, jour, note, titre):
    return {
        "id": identifiant,
        "statut": "TERMINEE",
        "nom_fichier": f"{identifiant}.pdf",
        "titre_cours": titre,
        "matiere": "Mathématiques",
        "niveau": "Dernière année du primaire",
        "date_creation": f"{jour:02d}/01/2026 09:00",
        "date_creation_iso": f"2026-01-{jour:02d}T09:00:00",
        "resume_note_globale": note,
    }


CORPUS = [
    _analyse("a", 1, 40, "Zeta"),
    _analyse("b", 15, 90, "Alpha"),
    _analyse("c", 8, 65, "Mu"),
]


class TestTri:

    def test_plus_recentes_d_abord_par_defaut(self):
        assert [a["id"] for a in _presentation.trier_analyses(CORPUS, "recent")] == ["b", "c", "a"]

    def test_plus_anciennes_d_abord(self):
        assert [a["id"] for a in _presentation.trier_analyses(CORPUS, "ancien")] == ["a", "c", "b"]

    def test_par_note(self):
        assert [a["id"] for a in _presentation.trier_analyses(CORPUS, "note_desc")] == ["b", "c", "a"]
        assert [a["id"] for a in _presentation.trier_analyses(CORPUS, "note_asc")] == ["a", "c", "b"]

    def test_par_nom(self):
        assert [a["id"] for a in _presentation.trier_analyses(CORPUS, "nom")] == ["b", "c", "a"]

    def test_un_tri_inconnu_retombe_sur_le_defaut(self):
        """Une clé venue d'une URL modifiée à la main ne doit pas lever."""
        attendu = _presentation.trier_analyses(CORPUS, _presentation.TRI_PAR_DEFAUT)
        assert _presentation.trier_analyses(CORPUS, "n_importe_quoi") == attendu

    def test_une_note_absente_ne_casse_pas_le_tri(self):
        corpus = CORPUS + [{"id": "sans_note", "statut": "TERMINEE"}]
        ordonnees = _presentation.trier_analyses(corpus, "note_desc")
        assert ordonnees[-1]["id"] == "sans_note"

    def test_le_tri_ne_perd_aucune_analyse(self):
        for cle in _presentation.TRIS_ANALYSES:
            assert len(_presentation.trier_analyses(CORPUS, cle)) == len(CORPUS)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

class TestPagination:

    def test_decoupage(self):
        page = _presentation.paginer(list(range(30)), 2, taille=12)
        assert page["elements"] == list(range(12, 24))
        assert (page["page"], page["nb_pages"]) == (2, 3)
        assert (page["premier"], page["dernier"]) == (13, 24)

    def test_derniere_page_incomplete(self):
        page = _presentation.paginer(list(range(25)), 3, taille=12)
        assert len(page["elements"]) == 1
        assert page["dernier"] == 25
        assert page["a_suivant"] is False

    def test_page_hors_bornes_ramenee_dans_l_intervalle(self):
        """Une page inexistante doit montrer la dernière, pas une page vide."""
        assert _presentation.paginer(list(range(30)), 99, taille=12)["page"] == 3
        assert _presentation.paginer(list(range(30)), -5, taille=12)["page"] == 1

    def test_liste_vide(self):
        page = _presentation.paginer([], 1, taille=12)
        assert page["elements"] == []
        assert page["nb_pages"] == 1
        assert page["paginee"] is False
        assert (page["premier"], page["dernier"]) == (0, 0)

    def test_pas_de_pagination_en_dessous_du_seuil(self):
        assert _presentation.paginer(list(range(5)), 1, taille=12)["paginee"] is False

    def test_numero_de_page_aberrant_dans_l_url(self):
        assert _presentation.page_demandee({"page": "abc"}) == 1
        assert _presentation.page_demandee({}) == 1
        assert _presentation.page_demandee({"page": "3"}) == 3


# ---------------------------------------------------------------------------
# Historique : tri et pagination de bout en bout
# ---------------------------------------------------------------------------

class TestHistoriqueHttp:

    def _peupler(self, utilisateur_id, nombre):
        for i in range(nombre):
            database.creer_analyse(_analyse(f"h{i:02d}", (i % 28) + 1, 40 + i,
                                            f"Document {i:02d}"))
            database.mettre_a_jour_analyse(f"h{i:02d}", {"utilisateur_id": utilisateur_id})

    def test_la_pagination_apparait_au_dela_du_seuil(self, client, utilisateur_figee,
                                                     connecter):
        connecter(utilisateur_figee)
        self._peupler(utilisateur_figee["id"], _presentation.TAILLE_PAGE + 3)

        corps = client.get("/espace/historique").get_data(as_text=True)
        assert "Page 1 sur 2" in corps

    def test_la_pagination_conserve_le_tri_et_les_filtres(self, client,
                                                          utilisateur_figee, connecter):
        """
        Perdre son filtrage en changeant de page est la façon la plus sûre de
        rendre un historique inutilisable.
        """
        connecter(utilisateur_figee)
        self._peupler(utilisateur_figee["id"], _presentation.TAILLE_PAGE + 3)

        corps = client.get(
            "/espace/historique?tri=note_desc&matiere=Mathématiques"
        ).get_data(as_text=True)
        assert "tri=note_desc" in corps
        assert "page=2" in corps
        assert "matiere=Math" in corps

    def test_page_hors_bornes_ne_donne_pas_une_page_vide(self, client,
                                                         utilisateur_figee, connecter):
        connecter(utilisateur_figee)
        self._peupler(utilisateur_figee["id"], 5)
        reponse = client.get("/espace/historique?page=99")
        assert reponse.status_code == 200
        assert "Document 0" in reponse.get_data(as_text=True)

    def test_l_historique_admin_est_aussi_pagine(self, client, connecter):
        administrateur = {"id": "adm", "email": "a@b.c", "nom": "Admin",
                          "role": "administrateur", "actif": True}
        connecter(administrateur)
        self._peupler("qui_que_ce_soit", _presentation.TAILLE_PAGE + 3)

        corps = client.get("/administration/analyses").get_data(as_text=True)
        assert "Page 1 sur 2" in corps


# ---------------------------------------------------------------------------
# Filet de rendu : chaque page doit répondre 200
# ---------------------------------------------------------------------------

class TestRenduDesGabarits:
    """
    Une variable oubliée dans un `render_template` ne se voit qu'à l'ouverture
    de la page. Ce filet demande chaque écran et échoue au premier `pytest`.
    """

    PAGES_PUBLIQUES = ["/", "/connexion", "/architecture", "/api/referentiels"]

    PAGES_ENSEIGNANT = [
        "/espace/", "/espace/historique", "/espace/profil", "/programmes/",
    ]

    PAGES_ADMIN = [
        "/administration/", "/administration/utilisateurs",
        "/administration/analyses", "/administration/systeme",
        "/administration/modeles",
    ]

    @pytest.mark.parametrize("chemin", PAGES_PUBLIQUES)
    def test_pages_publiques(self, client, chemin):
        assert client.get(chemin).status_code == 200, chemin

    @pytest.mark.parametrize("chemin", PAGES_ENSEIGNANT)
    def test_pages_enseignant(self, client, utilisateur_figee, connecter, chemin):
        connecter(utilisateur_figee)
        assert client.get(chemin).status_code == 200, chemin

    @pytest.mark.parametrize("chemin", PAGES_ADMIN)
    def test_pages_administrateur(self, client, connecter, chemin):
        connecter({"id": "adm", "email": "a@b.c", "nom": "Admin",
                   "role": "administrateur", "actif": True})
        assert client.get(chemin).status_code == 200, chemin

    def test_page_d_erreur_404(self, client):
        reponse = client.get("/chemin/qui/n/existe/pas")
        assert reponse.status_code == 404
        corps = reponse.get_data(as_text=True)
        # La page d'erreur doit proposer une sortie, pas seulement un code.
        assert "accueil" in corps.lower()
