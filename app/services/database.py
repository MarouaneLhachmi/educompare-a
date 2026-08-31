"""
Couche base de donnees (MongoDB).
==================================

Conformement a la section 3.4.4 du rapport de conception, la persistance des
donnees applicatives (comptes utilisateurs, historique des analyses,
metadonnees des documents deposes, resultats des analyses) est assuree par
MongoDB, une base orientee documents.

Ce module expose :
- une connexion paresseuse a MongoDB (`get_db()`) ;
- un **repli en memoire** transparent (`InMemoryCollection`) implementant le
  sous-ensemble de l'API pymongo reellement utilise, afin que l'application
  reste demontrable meme si aucun serveur MongoDB n'est demarre ;
- des fonctions de depot (repositories) pour les trois collections du
  systeme : `utilisateurs`, `analyses` et `evenements`.
"""

import threading
import uuid
from collections import Counter
from datetime import datetime, timezone

from app.config import Config

_STATE = {
    "client": None,
    "db": None,
    "mode": None,  # "mongodb" | "memoire"
    "erreur": None,
}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Repli en memoire : mini-implementation compatible pymongo
# ---------------------------------------------------------------------------

def _match(document: dict, filtre: dict) -> bool:
    for cle, attendu in (filtre or {}).items():
        valeur = document.get(cle)
        if isinstance(attendu, dict):
            if "$in" in attendu and valeur not in attendu["$in"]:
                return False
            if "$ne" in attendu and valeur == attendu["$ne"]:
                return False
            if "$exists" in attendu and (valeur is not None) != attendu["$exists"]:
                return False
        elif valeur != attendu:
            return False
    return True


class _Cursor(list):
    """Curseur minimaliste supportant .sort() et .limit() en chaine."""

    def sort(self, cle, sens=1):
        return _Cursor(
            sorted(self, key=lambda d: (d.get(cle) is None, d.get(cle)), reverse=sens < 0)
        )

    def limit(self, n):
        return _Cursor(self[:n]) if n else self

    def skip(self, n):
        return _Cursor(self[n:])


class InMemoryCollection:
    """Collection en memoire (repli quand MongoDB n'est pas disponible)."""

    def __init__(self, nom: str):
        self.nom = nom
        self._docs: list[dict] = []
        self._lock = threading.Lock()

    def insert_one(self, document: dict):
        with self._lock:
            document.setdefault("_id", str(uuid.uuid4()))
            self._docs.append(dict(document))
        return type("Res", (), {"inserted_id": document["_id"]})()

    def find_one(self, filtre: dict | None = None):
        with self._lock:
            for doc in self._docs:
                if _match(doc, filtre or {}):
                    return dict(doc)
        return None

    def find(self, filtre: dict | None = None, projection=None):
        with self._lock:
            return _Cursor(dict(d) for d in self._docs if _match(d, filtre or {}))

    def update_one(self, filtre: dict, maj: dict, upsert: bool = False):
        with self._lock:
            for doc in self._docs:
                if _match(doc, filtre or {}):
                    doc.update(maj.get("$set", {}))
                    for cle, pas in (maj.get("$inc") or {}).items():
                        doc[cle] = (doc.get(cle) or 0) + pas
                    return type("Res", (), {"matched_count": 1, "modified_count": 1})()
            if upsert:
                nouveau = dict(filtre)
                nouveau.update(maj.get("$set", {}))
                nouveau.setdefault("_id", str(uuid.uuid4()))
                self._docs.append(nouveau)
                return type("Res", (), {"matched_count": 0, "modified_count": 0})()
        return type("Res", (), {"matched_count": 0, "modified_count": 0})()

    def delete_one(self, filtre: dict):
        with self._lock:
            for i, doc in enumerate(self._docs):
                if _match(doc, filtre or {}):
                    del self._docs[i]
                    return type("Res", (), {"deleted_count": 1})()
        return type("Res", (), {"deleted_count": 0})()

    def count_documents(self, filtre: dict | None = None) -> int:
        with self._lock:
            return sum(1 for d in self._docs if _match(d, filtre or {}))

    def create_index(self, *args, **kwargs):  # compatibilite d'interface
        return None


class InMemoryDB:
    def __init__(self):
        self._collections: dict[str, InMemoryCollection] = {}

    def __getitem__(self, nom: str) -> InMemoryCollection:
        if nom not in self._collections:
            self._collections[nom] = InMemoryCollection(nom)
        return self._collections[nom]

    def get_collection(self, nom):
        return self[nom]


# ---------------------------------------------------------------------------
# Connexion
# ---------------------------------------------------------------------------

def get_db():
    """Retourne la base de donnees (MongoDB si joignable, sinon repli memoire)."""
    if _STATE["db"] is not None:
        return _STATE["db"]
    with _LOCK:
        if _STATE["db"] is not None:
            return _STATE["db"]
        try:
            from pymongo import MongoClient

            client = MongoClient(
                Config.MONGO_URI, serverSelectionTimeoutMS=Config.MONGO_TIMEOUT_MS
            )
            client.admin.command("ping")  # verifie reellement la connexion
            db = client[Config.MONGO_DB_NAME]
            db["utilisateurs"].create_index("email", unique=True)
            db["analyses"].create_index("id")
            db["analyses"].create_index("utilisateur_id")
            _STATE.update({"client": client, "db": db, "mode": "mongodb", "erreur": None})
        except Exception as exc:
            _STATE.update(
                {
                    "client": None,
                    "db": InMemoryDB(),
                    "mode": "memoire",
                    "erreur": str(exc).split("\n")[0][:200],
                }
            )
    return _STATE["db"]


def statut_connexion() -> dict:
    get_db()
    return {
        "mode": _STATE["mode"],
        "uri": Config.MONGO_URI if _STATE["mode"] == "mongodb" else None,
        "base": Config.MONGO_DB_NAME,
        "erreur": _STATE["erreur"],
        "disponible": _STATE["mode"] == "mongodb",
    }


def _maintenant() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Depot : utilisateurs
# ---------------------------------------------------------------------------

def _collection_utilisateurs():
    return get_db()["utilisateurs"]


def enregistrer_ou_mettre_a_jour_utilisateur(profil: dict) -> dict:
    """
    Cree le compte a la premiere connexion Google, ou met a jour les
    informations de profil et la date de derniere connexion s'il existe deja.
    """
    email = (profil.get("email") or "").strip().lower()
    if not email:
        raise ValueError("Un compte utilisateur necessite une adresse e-mail.")

    collection = _collection_utilisateurs()
    existant = collection.find_one({"email": email})

    role = "administrateur" if email in Config.ADMIN_EMAILS else "utilisateur"

    if existant:
        maj = {
            "nom": profil.get("nom") or existant.get("nom"),
            "photo": profil.get("photo") or existant.get("photo"),
            "fournisseur": profil.get("fournisseur", existant.get("fournisseur")),
            "derniere_connexion": _maintenant(),
        }
        # Un administrateur declare dans ADMIN_EMAILS le reste ; une promotion
        # manuelle faite depuis le back-office n'est jamais ecrasee.
        if role == "administrateur":
            maj["role"] = "administrateur"
        collection.update_one({"email": email}, {"$set": maj, "$inc": {"nb_connexions": 1}})
        utilisateur = collection.find_one({"email": email})
    else:
        utilisateur = {
            "_id": str(uuid.uuid4()),
            "id": str(uuid.uuid4())[:12],
            "email": email,
            "nom": profil.get("nom") or email.split("@")[0],
            "photo": profil.get("photo"),
            "fournisseur": profil.get("fournisseur", "google"),
            "role": role,
            "actif": True,
            "date_creation": _maintenant(),
            "derniere_connexion": _maintenant(),
            "nb_connexions": 1,
        }
        collection.insert_one(utilisateur)
        journaliser("compte_cree", utilisateur["id"], {"email": email})

    return _normaliser_utilisateur(utilisateur)


def _normaliser_utilisateur(utilisateur: dict | None) -> dict | None:
    if not utilisateur:
        return None
    utilisateur = dict(utilisateur)
    utilisateur["_id"] = str(utilisateur.get("_id"))
    utilisateur.setdefault("id", utilisateur["_id"][:12])
    utilisateur.setdefault("role", "utilisateur")
    utilisateur.setdefault("actif", True)
    return utilisateur


def utilisateur_par_id(utilisateur_id: str) -> dict | None:
    if not utilisateur_id:
        return None
    return _normaliser_utilisateur(_collection_utilisateurs().find_one({"id": utilisateur_id}))


def utilisateur_par_email(email: str) -> dict | None:
    return _normaliser_utilisateur(
        _collection_utilisateurs().find_one({"email": (email or "").strip().lower()})
    )


def lister_utilisateurs() -> list[dict]:
    curseur = _collection_utilisateurs().find({})
    utilisateurs = [_normaliser_utilisateur(u) for u in curseur]
    utilisateurs.sort(key=lambda u: str(u.get("date_creation") or ""), reverse=True)
    return utilisateurs


def definir_role(utilisateur_id: str, role: str) -> bool:
    if role not in {"utilisateur", "administrateur"}:
        raise ValueError("Role invalide.")
    resultat = _collection_utilisateurs().update_one(
        {"id": utilisateur_id}, {"$set": {"role": role}}
    )
    journaliser("role_modifie", utilisateur_id, {"role": role})
    return bool(resultat.matched_count)


def definir_activation(utilisateur_id: str, actif: bool) -> bool:
    resultat = _collection_utilisateurs().update_one(
        {"id": utilisateur_id}, {"$set": {"actif": bool(actif)}}
    )
    journaliser("compte_active" if actif else "compte_desactive", utilisateur_id, {})
    return bool(resultat.matched_count)


# ---------------------------------------------------------------------------
# Depot : analyses
# ---------------------------------------------------------------------------

def _collection_analyses():
    return get_db()["analyses"]


def creer_analyse(analyse: dict) -> dict:
    document = dict(analyse)
    document.setdefault("_id", str(uuid.uuid4()))
    _collection_analyses().insert_one(document)
    return document


def mettre_a_jour_analyse(analyse_id: str, champs: dict) -> None:
    _collection_analyses().update_one({"id": analyse_id}, {"$set": champs})


def enregistrer_analyse(analyse: dict) -> None:
    """Ecrit l'integralite d'une analyse (creation ou remplacement)."""
    champs = {k: v for k, v in analyse.items() if k != "_id"}
    resultat = _collection_analyses().update_one(
        {"id": analyse["id"]}, {"$set": champs}, upsert=True
    )
    if not resultat.matched_count:
        # upsert deja gere par update_one cote MongoDB comme en memoire
        pass


def analyse_par_id(analyse_id: str) -> dict | None:
    document = _collection_analyses().find_one({"id": analyse_id})
    if document:
        document = dict(document)
        document["_id"] = str(document.get("_id"))
    return document


def lister_analyses(utilisateur_id: str | None = None, limite: int | None = None) -> list[dict]:
    filtre = {"utilisateur_id": utilisateur_id} if utilisateur_id else {}
    documents = [dict(d) for d in _collection_analyses().find(filtre)]
    for d in documents:
        d["_id"] = str(d.get("_id"))
    documents.sort(key=lambda d: str(d.get("date_creation_iso") or ""), reverse=True)
    return documents[:limite] if limite else documents


def supprimer_analyse(analyse_id: str) -> bool:
    return bool(_collection_analyses().delete_one({"id": analyse_id}).deleted_count)


def compter_analyses(filtre: dict | None = None) -> int:
    return _collection_analyses().count_documents(filtre or {})


# ---------------------------------------------------------------------------
# Depot : journal d'evenements (tracabilite / supervision)
# ---------------------------------------------------------------------------

def journaliser(type_evenement: str, utilisateur_id: str | None, details: dict | None = None) -> None:
    try:
        get_db()["evenements"].insert_one(
            {
                "_id": str(uuid.uuid4()),
                "type": type_evenement,
                "utilisateur_id": utilisateur_id,
                "details": details or {},
                "horodatage": _maintenant(),
            }
        )
    except Exception:
        # La tracabilite ne doit jamais interrompre le parcours utilisateur.
        pass


def lister_evenements(limite: int = 50) -> list[dict]:
    documents = [dict(d) for d in get_db()["evenements"].find({})]
    documents.sort(key=lambda d: str(d.get("horodatage") or ""), reverse=True)
    return documents[:limite]


# ---------------------------------------------------------------------------
# Depot : retours enseignants (boucle de retour, phase 1.2 du plan de
# transition — mode ombre : on collecte, le comportement du systeme ne change
# pas encore)
# ---------------------------------------------------------------------------

TYPES_RETOUR = {"couverture_notion", "pertinence_recommandation", "qualite_exercice"}


def _collection_retours():
    return get_db()["retours"]


def enregistrer_retour(analyse_id: str, type_retour: str, cle_notion: str,
                       valeur: str, utilisateur_id: str | None) -> dict:
    """
    Enregistre une etiquette produite par un enseignant en train de lire son
    rapport. `cle_notion` identifie ce sur quoi porte le retour : une notion
    du referentiel (« FR::Fractions... »), une etape de parcours (« rang:3 »)
    ou un exercice (« rang:3::exercice:1 ») selon `type_retour`.

    Aucune verification de coherence n'est faite ici au-dela du type : c'est
    la matiere premiere brute du retour, pas encore une decision.
    """
    if type_retour not in TYPES_RETOUR:
        raise ValueError(f"Type de retour inconnu : {type_retour}")
    if not cle_notion:
        raise ValueError("Un retour doit porter sur une cible identifiee.")

    document = {
        "_id": str(uuid.uuid4()),
        "analyse_id": analyse_id,
        "type": type_retour,
        "cle_notion": cle_notion,
        "valeur": valeur,
        "utilisateur_id": utilisateur_id,
        "horodatage": _maintenant(),
    }
    _collection_retours().insert_one(document)
    return document


def lister_retours(analyse_id: str | None = None) -> list[dict]:
    filtre = {"analyse_id": analyse_id} if analyse_id else {}
    documents = [dict(d) for d in _collection_retours().find(filtre)]
    documents.sort(key=lambda d: str(d.get("horodatage") or ""), reverse=True)
    return documents


def compter_retours() -> dict:
    """Volume accumule, pour la supervision : c'est le seul effet visible
    du mode ombre tant que la bascule (~300 etiquettes) n'est pas activee."""
    documents = list(_collection_retours().find({}))
    par_type = Counter(d.get("type") for d in documents)
    return {
        "total": len(documents),
        "par_type": {t: par_type.get(t, 0) for t in sorted(TYPES_RETOUR)},
    }


# ---------------------------------------------------------------------------
# Depot : programmes (du cours au cursus, phase 2.1 du plan de transition)
# ---------------------------------------------------------------------------
#
# Un programme regroupe les analyses des documents d'un meme cursus — les
# chapitres d'une annee, repartis entre plusieurs enseignants. Il ne duplique
# aucune donnee d'analyse : il n'en tient que la liste, et la couverture est
# recalculee a la volee par l'Agent 6 en mode agrege.

def _collection_programmes():
    return get_db()["programmes"]


def creer_programme(programme: dict) -> dict:
    document = dict(programme)
    document.setdefault("_id", str(uuid.uuid4()))
    document.setdefault("id", uuid.uuid4().hex[:10])
    document.setdefault("analyse_ids", [])
    document.setdefault("date_creation", _maintenant())
    _collection_programmes().insert_one(document)
    return document


def programme_par_id(programme_id: str) -> dict | None:
    document = _collection_programmes().find_one({"id": programme_id})
    if document:
        document = dict(document)
        document["_id"] = str(document.get("_id"))
    return document


def lister_programmes(utilisateur_id: str | None = None) -> list[dict]:
    filtre = {"utilisateur_id": utilisateur_id} if utilisateur_id else {}
    documents = [dict(d) for d in _collection_programmes().find(filtre)]
    for d in documents:
        d["_id"] = str(d.get("_id"))
    documents.sort(key=lambda d: str(d.get("date_creation") or ""), reverse=True)
    return documents


def mettre_a_jour_programme(programme_id: str, champs: dict) -> None:
    _collection_programmes().update_one({"id": programme_id}, {"$set": champs})


def supprimer_programme(programme_id: str) -> bool:
    """Ne supprime que le regroupement : les analyses lui survivent."""
    return bool(_collection_programmes().delete_one({"id": programme_id}).deleted_count)


def rattacher_analyse(programme_id: str, analyse_id: str) -> bool:
    programme = programme_par_id(programme_id)
    if programme is None:
        return False
    identifiants = list(programme.get("analyse_ids") or [])
    if analyse_id in identifiants:
        return False
    identifiants.append(analyse_id)
    mettre_a_jour_programme(programme_id, {"analyse_ids": identifiants})
    return True


def detacher_analyse(programme_id: str, analyse_id: str) -> bool:
    programme = programme_par_id(programme_id)
    if programme is None:
        return False
    identifiants = [i for i in (programme.get("analyse_ids") or []) if i != analyse_id]
    if len(identifiants) == len(programme.get("analyse_ids") or []):
        return False
    mettre_a_jour_programme(programme_id, {"analyse_ids": identifiants})
    return True


def analyses_du_programme(programme: dict) -> list[dict]:
    """
    Analyses rattachees, dans l'ordre du programme. Les identifiants
    orphelins — analyse supprimee depuis — sont ignores silencieusement ici ;
    c'est la couche d'affichage qui les signale.
    """
    resultat = []
    for analyse_id in (programme.get("analyse_ids") or []):
        analyse = analyse_par_id(analyse_id)
        if analyse is not None:
            resultat.append(analyse)
    return resultat
