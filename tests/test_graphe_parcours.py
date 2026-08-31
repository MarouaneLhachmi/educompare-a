"""
Tests de l'ordonnancement par centralite de graphe (Agent 8, source 2).
=======================================================================

Cet algorithme existe pour une raison precise : produire un ordre de priorite
qui puisse **contredire** celui du planificateur par renforcement. S'il rendait
toujours le meme verdict, il n'apporterait rien et la confrontation a trois
voix serait un decor.

Les tests portent donc autant sur ce qu'il calcule que sur ce qui le
distingue :

- la centralite remonte-t-elle vers les notions fondatrices ?
- l'ordre respecte-t-il les prerequis, meme quand le score dit l'inverse ?
- diverge-t-il reellement d'un classement par gravite — c'est-a-dire de ce que
  l'Agent 6 produirait deja tout seul ?
- ne bloque-t-il jamais, quelles que soient les donnees ?
"""

import pytest

from app.services import graphe_parcours, rl_parcours


def _notion(cle: str, *, gravite: float = 0.8, maitrise: float = 0.2,
            consensus: float = 0.5, type_ecart: str = "absente") -> dict:
    return {
        "cle": cle, "notion": cle, "descriptif": f"Descriptif de {cle}",
        "pays": "France", "code": "FR",
        "maitrise": maitrise, "consensus": consensus, "gravite": gravite,
        "type_ecart": type_ecart, "libelle_ecart": type_ecart,
    }


def _graphe(prerequis: dict) -> dict:
    return {
        "prerequis": prerequis,
        "methode": "graphe de test",
        "nb_aretes": sum(len(v) for v in prerequis.values()),
    }


# ---------------------------------------------------------------------------
# Centralite
# ---------------------------------------------------------------------------

class TestCentralite:

    def test_la_notion_fondatrice_obtient_la_centralite_maximale(self):
        """
        Sur une chaine A -> B -> C -> D, A conditionne tout le reste : c'est
        elle qui doit ressortir, pas la derniere du programme.
        """
        etat = [_notion(c) for c in "ABCD"]
        graphe = _graphe({"A": [], "B": ["A"], "C": ["B"], "D": ["C"]})

        parcours = graphe_parcours.planifier(etat, graphe)
        par_notion = {e["notion"]: e for e in parcours["etapes"]}

        assert par_notion["A"]["centralite"] == 1.0
        assert par_notion["A"]["centralite"] > par_notion["D"]["centralite"]

    def test_le_nombre_de_notions_dependantes_est_compte(self):
        etat = [_notion(c) for c in ["Pivot", "X", "Y", "Z"]]
        graphe = _graphe({"Pivot": [], "X": ["Pivot"], "Y": ["Pivot"], "Z": ["Pivot"]})

        parcours = graphe_parcours.planifier(etat, graphe)
        par_notion = {e["notion"]: e for e in parcours["etapes"]}

        assert par_notion["Pivot"]["nb_notions_dependantes"] == 3
        assert par_notion["X"]["nb_notions_dependantes"] == 0

    def test_une_notion_maitrisee_est_ecartee(self):
        """Prioriser ce qui est deja su reviendrait a faire reviser l'acquis."""
        etat = [_notion("Acquise", maitrise=0.95), _notion("Lacune", maitrise=0.10)]
        parcours = graphe_parcours.planifier(etat, _graphe({"Acquise": [], "Lacune": []}))

        assert [e["notion"] for e in parcours["etapes"]] == ["Lacune"]

    def test_le_nombre_d_etapes_est_plafonne(self):
        etat = [_notion(f"N{i}") for i in range(30)]
        parcours = graphe_parcours.planifier(etat, _graphe({f"N{i}": [] for i in range(30)}),
                                             max_etapes=5)
        assert parcours["nb_etapes"] == 5


# ---------------------------------------------------------------------------
# Ordonnancement
# ---------------------------------------------------------------------------

class TestOrdonnancement:

    def test_un_prerequis_retenu_passe_toujours_avant_sa_notion(self):
        """
        Le score dit QUOI traiter, le graphe dit DANS QUEL ORDRE. Ici « Suite »
        a la gravite la plus forte, mais elle depend de « Base » : la traiter
        d'abord serait pedagogiquement absurde.
        """
        etat = [_notion("Base", gravite=0.40), _notion("Suite", gravite=0.99)]
        graphe = _graphe({"Base": [], "Suite": ["Base"]})

        ordre = [e["notion"] for e in graphe_parcours.planifier(etat, graphe)["etapes"]]
        assert ordre.index("Base") < ordre.index("Suite")

    def test_les_rangs_sont_consecutifs_et_commencent_a_un(self):
        etat = [_notion(c) for c in "ABCD"]
        etapes = graphe_parcours.planifier(etat, _graphe({c: [] for c in "ABCD"}))["etapes"]
        assert [e["rang"] for e in etapes] == [1, 2, 3, 4]

    def test_les_seances_sont_cumulees(self):
        etat = [_notion(c) for c in "ABC"]
        etapes = graphe_parcours.planifier(etat, _graphe({c: [] for c in "ABC"}))["etapes"]
        cumuls = [e["seance_cumulee"] for e in etapes]
        assert cumuls == sorted(cumuls)
        assert cumuls[-1] > 0


# ---------------------------------------------------------------------------
# Complementarite — la raison d'etre de cet algorithme
# ---------------------------------------------------------------------------

class TestComplementarite:

    def test_il_diverge_d_un_classement_par_gravite(self):
        """
        Le test central. « Pivot » a un ecart modere mais quatre notions en
        dependent ; « Isolee » a l'ecart le plus severe mais ne conditionne
        rien. Un classement par gravite — ce que l'Agent 6 produit deja —
        commencerait par « Isolee ». Si cet algorithme faisait de meme, il
        n'apporterait aucune information nouvelle.
        """
        etat = (
            [_notion("Pivot", gravite=0.45)]
            + [_notion(f"Aval{i}", gravite=0.30) for i in range(4)]
            + [_notion("Isolee", gravite=0.95)]
        )
        graphe = _graphe(
            {"Pivot": [], "Isolee": []} | {f"Aval{i}": ["Pivot"] for i in range(4)}
        )

        parcours = graphe_parcours.planifier(etat, graphe)
        premier_graphe = parcours["etapes"][0]["notion"]
        premier_gravite = max(etat, key=lambda n: n["gravite"])["notion"]

        assert premier_graphe == "Pivot"
        assert premier_gravite == "Isolee"
        assert premier_graphe != premier_gravite

    def test_il_n_est_pas_une_variante_du_planificateur(self):
        """
        Les deux algorithmes tournent sur le MEME etat et le MEME graphe. S'ils
        rendaient le meme ordre, la confrontation a trois voix n'aurait aucun
        contenu.
        """
        etat = (
            [_notion("Pivot", gravite=0.45, maitrise=0.30)]
            + [_notion(f"Aval{i}", gravite=0.30, maitrise=0.35) for i in range(4)]
            + [_notion("Isolee", gravite=0.95, maitrise=0.10)]
        )
        graphe = _graphe(
            {"Pivot": [], "Isolee": []} | {f"Aval{i}": ["Pivot"] for i in range(4)}
        )

        par_graphe = graphe_parcours.planifier(etat, graphe, max_etapes=6)
        par_rl = rl_parcours.planifier(etat, graphe, budget_seances=12.0, max_etapes=6)

        ordre_graphe = [e["notion"] for e in par_graphe["etapes"]]
        ordre_rl = [e["notion"] for e in par_rl["etapes"]]
        assert ordre_graphe != ordre_rl, (
            "les deux sources locales rendent le même ordre : la confrontation "
            "à trois voix serait un décor"
        )

    def test_il_expose_ce_que_le_planificateur_ne_calcule_pas(self):
        """Centralite et nombre de dependantes sont propres a cette source."""
        etat = [_notion(c) for c in "AB"]
        etape = graphe_parcours.planifier(etat, _graphe({"A": [], "B": ["A"]}))["etapes"][0]
        assert "centralite" in etape
        assert "nb_notions_dependantes" in etape
        assert "score_priorisation" in etape

    def test_il_reutilise_les_interventions_du_planificateur(self):
        """
        Les deux parcours doivent rester comparables : meme vocabulaire
        d'interventions, sinon la confrontation compare des choux et des
        carottes.
        """
        etat = [_notion("A", type_ecart="absente"), _notion("B", type_ecart="superficielle")]
        etapes = graphe_parcours.planifier(etat, _graphe({"A": [], "B": []}))["etapes"]
        connues = {i["cle"] for i in rl_parcours.INTERVENTIONS}
        for etape in etapes:
            assert etape["intervention"] in connues

    def test_l_algorithme_se_declare_non_supervise(self):
        infos = graphe_parcours.infos()
        assert "PageRank" in infos["algorithme"]
        assert "aucun" in infos["entrainement"].lower()
        assert infos["complementarite"]


# ---------------------------------------------------------------------------
# Robustesse — il ne doit jamais bloquer l'analyse
# ---------------------------------------------------------------------------

class TestRobustesse:

    def test_aucune_notion(self):
        parcours = graphe_parcours.planifier([], _graphe({}))
        assert parcours["disponible"] is False
        assert parcours["motif"]
        assert parcours["etapes"] == []

    def test_graphe_vide_reste_exploitable(self):
        """
        Sans arete, la centralite est uniforme : l'ordre retombe sur la
        gravite. C'est un repli, pas une panne.
        """
        etat = [_notion(c) for c in "ABC"]
        parcours = graphe_parcours.planifier(etat, {})
        assert parcours["disponible"] is True
        assert parcours["nb_etapes"] == 3

    def test_toutes_les_notions_acquises(self):
        etat = [_notion(c, maitrise=0.95) for c in "ABC"]
        parcours = graphe_parcours.planifier(etat, _graphe({c: [] for c in "ABC"}))
        assert parcours["disponible"] is False
        assert "écart" in parcours["motif"]

    def test_prerequis_pointant_hors_du_perimetre(self):
        """Un prerequis absent de l'etat ne doit pas faire echouer le calcul."""
        etat = [_notion("A")]
        parcours = graphe_parcours.planifier(etat, _graphe({"A": ["Inconnue"]}))
        assert parcours["disponible"] is True
        assert parcours["etapes"][0]["notion"] == "A"

    def test_cycle_dans_le_graphe_ne_bloque_pas(self):
        """
        Le graphe est acyclique par construction, mais l'ordonnancement ne doit
        pas boucler indefiniment si cette garantie tombait un jour.
        """
        etat = [_notion(c) for c in "AB"]
        parcours = graphe_parcours.planifier(etat, _graphe({"A": ["B"], "B": ["A"]}))
        assert parcours["disponible"] is True
        assert parcours["nb_etapes"] == 2

    def test_une_notion_unique(self):
        parcours = graphe_parcours.planifier([_notion("Seule")], _graphe({"Seule": []}))
        assert parcours["nb_etapes"] == 1

    def test_le_resultat_est_reproductible(self):
        etat = [_notion(c) for c in "ABCDE"]
        graphe = _graphe({"A": [], "B": ["A"], "C": ["A"], "D": ["B"], "E": ["C"]})
        premier = graphe_parcours.planifier(etat, graphe)
        second = graphe_parcours.planifier(etat, graphe)
        assert [e["notion"] for e in premier["etapes"]] == \
            [e["notion"] for e in second["etapes"]]


# ---------------------------------------------------------------------------
# PageRank lui-meme
# ---------------------------------------------------------------------------

class TestPageRank:

    def test_les_scores_somment_a_un(self):
        import numpy as np

        adjacence = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        scores = graphe_parcours._pagerank(adjacence, np.ones(3))
        assert scores.sum() == pytest.approx(1.0, abs=1e-6)

    def test_graphe_sans_arete_donne_des_scores_uniformes(self):
        import numpy as np

        scores = graphe_parcours._pagerank(np.zeros((4, 4)), np.ones(4))
        assert scores == pytest.approx([0.25] * 4, abs=1e-6)

    def test_la_personnalisation_oriente_le_score(self):
        """
        Sans arete, le score doit suivre exactement la personnalisation : c'est
        elle qui porte le besoin pedagogique.
        """
        import numpy as np

        scores = graphe_parcours._pagerank(np.zeros((2, 2)), np.array([3.0, 1.0]))
        assert scores[0] > scores[1]

    def test_graphe_vide(self):
        import numpy as np

        assert len(graphe_parcours._pagerank(np.zeros((0, 0)), np.zeros(0))) == 0


# ---------------------------------------------------------------------------
# Confrontation a trois voix (Agent 8)
# ---------------------------------------------------------------------------

class TestConfrontationTroisVoix:
    """
    La confrontation doit distinguer trois situations : l'unanimite des
    sources actives, l'accord partiel, et la notion isolee. C'est cette
    distinction qui donne sa valeur a la separation des sources.
    """

    @staticmethod
    def _parcours(notions: list[str]) -> dict:
        return {
            "disponible": True,
            "etapes": [
                {"rang": i, "notion": n, "cle": f"FR::{n}",
                 "intervention_nom": "Apport théorique", "consensus": 0.5,
                 "centralite": 0.8, "nb_notions_dependantes": 2}
                for i, n in enumerate(notions, start=1)
            ],
        }

    @staticmethod
    def _libres(notions: list[str]) -> dict:
        return {
            "disponible": True,
            "recommandations": [
                {"titre": f"Renforcer {n}", "notion_visee": n, "priorite": "Haute"}
                for n in notions
            ],
        }

    def test_une_notion_vue_par_les_trois_sources_fait_l_unanimite(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions", "Géométrie"]),
            self._parcours(["Fractions", "Mesures"]),
            self._libres(["Fractions"]),
        )
        assert c["disponible"] is True
        assert c["nb_sources"] == 3
        unanimes = [e["notion"] for e in c["accords_complets"]]
        assert unanimes == ["Fractions"]
        assert c["accords_complets"][0]["nb_sources"] == 3

    def test_un_accord_entre_deux_sources_sur_trois_est_partiel(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions"]),
            self._parcours(["Fractions"]),
            self._libres(["Autre chose entièrement"]),
        )
        partiels = [e["notion"] for e in c["accords_partiels"]]
        assert "Fractions" in partiels
        assert c["accords_complets"] == []

    def test_une_notion_vue_par_une_seule_source_est_isolee(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions"]),
            self._parcours(["Mesures"]),
            {"disponible": False},
        )
        isolees = sorted(e["notion"] for e in c["isolees"])
        assert isolees == ["Fractions", "Mesures"]

    def test_l_ecart_de_rang_entre_les_deux_algorithmes_est_mesure(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["A", "B", "Commune"]),
            self._parcours(["Commune", "C", "D"]),
            {"disponible": False},
        )
        # « Commune » est au rang 3 pour l'un, au rang 1 pour l'autre.
        assert c["ecart_moyen_de_rang_entre_algorithmes"] == 2.0

    def test_le_gemini_indisponible_laisse_deux_sources_actives(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions"]), self._parcours(["Fractions"]),
            {"disponible": False},
        )
        assert c["disponible"] is True
        assert c["nb_sources"] == 2
        assert c["sources"] == {"planificateur": True, "graphe": True, "gemini": False}
        # Avec deux sources, l'unanimite porte sur deux voix.
        assert [e["notion"] for e in c["accords_complets"]] == ["Fractions"]

    def test_une_seule_source_rend_la_confrontation_impossible(self):
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions"]),
            {"disponible": False, "etapes": []},
            {"disponible": False},
        )
        assert c["disponible"] is False
        assert "moins de deux sources" in c["lecture"].lower()

    def test_les_cles_historiques_restent_alimentees(self):
        """
        L'export PDF consomme `convergences`, `specifiques_algorithme` et
        `taux_convergence_pct` : passer a trois voix ne doit pas les retirer.
        """
        from app.agents import agent8_recommandations as a8

        c = a8._confronter(
            self._parcours(["Fractions", "Géométrie"]),
            self._parcours(["Fractions"]),
            self._libres(["Fractions"]),
        )
        assert "convergences" in c and "specifiques_algorithme" in c
        assert "specifiques_gemini" in c and "taux_convergence_pct" in c
        assert any(x["notion"] == "Fractions" for x in c["convergences"])
