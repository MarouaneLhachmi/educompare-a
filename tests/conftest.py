"""
Socle commun des tests EduCompare AI.
======================================

Deux exigences guident ce fichier :

**Les tests ne doivent jamais sortir de la machine.** Aucun appel a l'API
Gemini, aucune connexion MongoDB. Les deux sont neutralises par des fixtures
`autouse`, donc sans que chaque test ait a y penser. Un test qui veut
justement verifier le comportement *avec* modele de langage demande
explicitement la fixture `llm_simule`.

**Les tests ne doivent rien laisser derriere eux.** La base est reinitialisee
avant chaque test, et les fichiers produits vont dans le dossier temporaire de
pytest, jamais dans `uploads/` ni `outputs/`.

Neutraliser le LLM par defaut n'est pas un contournement : c'est le mode
degrade que le systeme revendique. Chaque agent generatif dispose d'un repli
local, et les tests verifient d'abord que ce repli produit bien un resultat
complet — c'est la propriete la plus importante a proteger.
"""

import json
import os
import sys

import pytest

# Le dossier `tests/` est ajoute au sys.path par pytest, pas la racine du
# projet : on l'ajoute explicitement pour que `import app` fonctionne meme
# sans passer par le fichier de configuration.
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

# Neutralisation avant tout import de `app.config` : les valeurs sont lues au
# chargement de la classe Config.
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("FLASK_SECRET_KEY", "cle-de-test")
os.environ.setdefault("ALLOW_DEMO_LOGIN", "true")
os.environ.setdefault("MONGO_TIMEOUT_MS", "200")

from app.services import database, gemini_client  # noqa: E402

DOSSIER_TESTS = os.path.dirname(os.path.abspath(__file__))
DOSSIER_CORPUS = os.path.join(DOSSIER_TESTS, "corpus_reference")
CATALOGUE_PATH = os.path.join(DOSSIER_CORPUS, "catalogue.json")
ANCRAGES_PATH = os.path.join(DOSSIER_TESTS, "ancrages.json")


# ---------------------------------------------------------------------------
# Isolation : base de donnees
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def base_memoire():
    """
    Force le repli en memoire de `services.database` et le vide avant chaque
    test. On court-circuite volontairement `get_db()` : laisser le module
    tenter une connexion MongoDB ferait dependre la duree des tests de la
    presence d'un serveur local.
    """
    etat_initial = dict(database._STATE)
    database._STATE.update(
        {"client": None, "db": database.InMemoryDB(), "mode": "memoire",
         "erreur": "base en memoire (tests)"}
    )
    yield database._STATE["db"]
    database._STATE.clear()
    database._STATE.update(etat_initial)


# ---------------------------------------------------------------------------
# Isolation : modele de langage
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def llm_hors_ligne(monkeypatch):
    """
    Coupe l'acces au modele de langage pour tous les tests. Les agents
    generatifs empruntent donc systematiquement leur repli local.
    """
    def _indisponible(*args, **kwargs):
        raise gemini_client.LLMUnavailable("Modele de langage neutralise (tests).")

    monkeypatch.setattr(gemini_client, "generate_text", _indisponible)
    monkeypatch.setattr(gemini_client, "generate_json", _indisponible)
    monkeypatch.setattr(gemini_client, "is_configured", lambda: False)


class LLMSimule:
    """
    Faux client de modele de langage. On lui depose des reponses, il les rend
    dans l'ordre et enregistre les invites recues, ce qui permet de tester le
    chemin nominal des agents generatifs sans reseau.
    """

    def __init__(self):
        self.reponses: list = []
        self.invites: list[str] = []

    def repondre(self, *reponses) -> "LLMSimule":
        self.reponses.extend(reponses)
        return self

    def _suivante(self, prompt: str):
        self.invites.append(prompt)
        if not self.reponses:
            raise gemini_client.LLMUnavailable("Aucune reponse simulee restante.")
        return self.reponses.pop(0)

    def generate_text(self, prompt: str, agent: str = "inconnu", **kwargs) -> str:
        reponse = self._suivante(prompt)
        return reponse if isinstance(reponse, str) else json.dumps(reponse, ensure_ascii=False)

    def generate_json(self, prompt: str, agent: str = "inconnu", **kwargs):
        reponse = self._suivante(prompt)
        return gemini_client.extract_json(reponse) if isinstance(reponse, str) else reponse


@pytest.fixture
def llm_simule(monkeypatch):
    """
    Remplace le client Gemini par `LLMSimule`. Demander cette fixture annule
    la neutralisation posee par `llm_hors_ligne` (elle s'applique apres, les
    fixtures explicites etant resolues apres les fixtures `autouse`).
    """
    faux = LLMSimule()
    monkeypatch.setattr(gemini_client, "generate_text", faux.generate_text)
    monkeypatch.setattr(gemini_client, "generate_json", faux.generate_json)
    monkeypatch.setattr(gemini_client, "is_configured", lambda: True)
    return faux


# ---------------------------------------------------------------------------
# Corpus de reference
# ---------------------------------------------------------------------------

def charger_catalogue() -> list[dict]:
    """Catalogue du corpus de reference (genere par `generer_corpus.py`)."""
    if not os.path.exists(CATALOGUE_PATH):
        pytest.skip(
            "Corpus de reference absent : lancer "
            "`python tests/corpus_reference/generer_corpus.py`."
        )
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


def chemin_corpus(nom_fichier: str) -> str:
    return os.path.join(DOSSIER_CORPUS, nom_fichier)


def charger_ancrages() -> dict:
    """Intervalles d'ancrage mesures (voir `tests/mesurer_ancrages.py`)."""
    if not os.path.exists(ANCRAGES_PATH):
        return {}
    with open(ANCRAGES_PATH, "r", encoding="utf-8") as fichier:
        return json.load(fichier)


@pytest.fixture(scope="session")
def catalogue() -> list[dict]:
    return charger_catalogue()


@pytest.fixture(scope="session")
def ancrages() -> dict:
    return charger_ancrages()


@pytest.fixture
def cours_complet() -> str:
    """Chemin du support de cours complet de reference."""
    return chemin_corpus("cours_maths_complet.pdf")


# ---------------------------------------------------------------------------
# Entrees figees pour les tests unitaires d'agents
# ---------------------------------------------------------------------------
# Ces structures reproduisent la forme exacte des sorties d'agents, sans avoir
# a executer la chaine amont. C'est ce qui permet de tester un agent seul.

@pytest.fixture
def agent1_figee() -> dict:
    texte = (
        "Chapitre 1 : Les fractions. L'eleve apprend a comparer des fractions de meme "
        "denominateur et a calculer un pourcentage. "
        "Chapitre 2 : La geometrie. L'eleve doit tracer un carre et calculer son perimetre."
    )
    return {
        "nb_pages": 2,
        "nb_mots": len(texte.split()),
        "nb_caracteres": len(texte),
        "langue_detectee": "fr",
        "methode_extraction": "couche texte native du PDF (pypdf)",
        "ocr_utilise": False,
        "metadonnees": {},
        "chapitres": [
            {
                "titre": "Chapitre 1 : Les fractions",
                "extrait": "L'eleve apprend a comparer des fractions.",
                "contenu": (
                    "L'eleve apprend a comparer des fractions de meme denominateur. "
                    "Il decouvre les fractions equivalentes et leur simplification. "
                    "Il calcule ensuite un pourcentage d'une quantite donnee. "
                    "Des exercices d'application sont proposes en fin de chapitre."
                ),
                "page": 1,
                "nb_mots": 40,
            },
            {
                "titre": "Chapitre 2 : La geometrie",
                "extrait": "L'eleve doit tracer un carre.",
                "contenu": (
                    "L'eleve doit tracer un carre et un rectangle a la regle et a l'equerre. "
                    "Il calcule le perimetre puis l'aire de ces figures usuelles. "
                    "La symetrie axiale est introduite sur des figures simples."
                ),
                "page": 2,
                "nb_mots": 34,
            },
        ],
        "nb_chapitres": 2,
        "mots_cles": [{"mot": "fractions", "occurrences": 3, "frequence_pct": 4.0}],
        "elements_pedagogiques": {"exercices": 2, "objectifs": 0, "definitions": 0, "exemples": 0},
        "texte_complet": texte,
        "texte_brut_tronque": texte[:2000],
    }


@pytest.fixture
def agent2_figee() -> dict:
    return {
        "titre_cours": "Mathematiques - Derniere annee du primaire",
        "chapitres": [
            {"titre": "Chapitre 1 : Les fractions",
             "objectifs": ["Comparer des fractions", "Calculer un pourcentage"],
             "notions": ["fraction", "pourcentage"]},
            {"titre": "Chapitre 2 : La geometrie",
             "objectifs": ["Tracer un carre", "Calculer un perimetre"],
             "notions": ["carre", "perimetre"]},
        ],
        "notions_cles_globales": ["fraction", "pourcentage", "carre", "perimetre"],
        "resume": "Support couvrant les fractions et la geometrie plane.",
        "source": "repli_deterministe",
    }


@pytest.fixture
def notions_figees() -> list[dict]:
    """Petit referentiel figé : deux notions couvertes, une absente."""
    return [
        {
            "code": "FR", "pays": "France", "drapeau": "", "referentiel": "Programme de test",
            "notion": "Comparer des fractions de meme denominateur",
            "descriptif": "Ranger et comparer des fractions ayant le meme denominateur.",
            "texte": "Comparer des fractions de meme denominateur. Ranger et comparer des "
                     "fractions ayant le meme denominateur.",
        },
        {
            "code": "FR", "pays": "France", "drapeau": "", "referentiel": "Programme de test",
            "notion": "Calculer le perimetre d'un carre",
            "descriptif": "Calculer le perimetre des figures planes usuelles.",
            "texte": "Calculer le perimetre d'un carre. Calculer le perimetre des figures "
                     "planes usuelles.",
        },
        {
            "code": "UK", "pays": "Royaume-Uni", "drapeau": "", "referentiel": "Test curriculum",
            "notion": "Interpreter des diagrammes circulaires",
            "descriptif": "Read and construct pie charts to solve problems.",
            "texte": "Interpreter des diagrammes circulaires. Read and construct pie charts "
                     "to solve problems.",
        },
    ]


# ---------------------------------------------------------------------------
# Application Flask
# ---------------------------------------------------------------------------

@pytest.fixture
def application(monkeypatch, tmp_path):
    """
    Instance Flask de test. Le prechauffage de l'encodeur semantique est
    neutralise : il chargerait un modele de plusieurs centaines de mega-octets
    dans un fil de fond a chaque creation d'application.
    """
    import app as paquet_app
    from app.config import Config

    monkeypatch.setattr(paquet_app, "_prechauffer", lambda application: None)
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "OUTPUT_FOLDER", str(tmp_path / "outputs"))

    instance = paquet_app.create_app()
    instance.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    return instance


@pytest.fixture
def client(application):
    return application.test_client()


@pytest.fixture
def creer_utilisateur():
    """Cree un compte en base, sans ouvrir de session."""
    def _creer(utilisateur: dict) -> dict:
        database.get_db()["utilisateurs"].insert_one(dict(utilisateur))
        return utilisateur

    return _creer


@pytest.fixture
def connecter(client, creer_utilisateur):
    """
    Ouvre une session pour un utilisateur, sans passer par OAuth. Le compte
    est aussi cree en base : les vues rechargent l'utilisateur a chaque
    requete, un simple cookie ne suffirait pas.

    Un seul utilisateur est connecte a la fois — le dernier appel gagne.
    """
    def _connecter(utilisateur: dict) -> dict:
        if database.utilisateur_par_id(utilisateur["id"]) is None:
            creer_utilisateur(utilisateur)
        with client.session_transaction() as session:
            session["utilisateur_id"] = utilisateur["id"]
            session["utilisateur_role"] = utilisateur.get("role", "utilisateur")
        return utilisateur

    return _connecter


@pytest.fixture
def utilisateur_figee() -> dict:
    return {
        "id": "utilisateur01",
        "email": "enseignant@example.org",
        "nom": "Enseignant de test",
        "role": "utilisateur",
        "actif": True,
    }
