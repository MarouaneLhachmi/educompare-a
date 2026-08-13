"""
Client Gemini partage par les agents "generatifs".
==================================================

Le rapport de conception prevoit l'appel a un modele de langage pour les
taches de comprehension, de comparaison, d'evaluation, de recommandation et
de generation du rapport final (agents 2, 6, 7, 8 et 9). Ce module centralise
cet acces afin que :

- un seul client soit instancie pour toute l'application ;
- tous les agents beneficient du meme traitement d'erreur (l'application ne
  doit JAMAIS planter si l'API n'est pas joignable ou si le quota gratuit est
  epuise : chaque agent dispose d'un repli algorithmique local) ;
- les reponses JSON soient extraites de maniere tolerante (le modele ajoute
  parfois des balises markdown autour du JSON) ;
- les appels soient traces (latence, statut) pour alimenter les statistiques
  du tableau de bord administrateur.
"""

import json
import re
import time
import threading

from app.config import Config

_CLIENT = None
_CLIENT_LOCK = threading.Lock()
_CLIENT_ERROR: str | None = None

# Journal en memoire des appels LLM (alimente le dashboard administrateur).
CALL_LOG: list[dict] = []
_LOG_LOCK = threading.Lock()
_LOG_MAX = 200


class LLMUnavailable(Exception):
    """Levee quand l'API n'est pas configuree ou pas joignable."""


def is_configured() -> bool:
    return Config.gemini_configured()


def _get_client():
    """Instancie (une seule fois) le client google-genai."""
    global _CLIENT, _CLIENT_ERROR
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        if not is_configured():
            raise LLMUnavailable(
                "Cle API Gemini non configuree (variable GEMINI_API_KEY du fichier .env)."
            )
        try:
            from google import genai

            _CLIENT = genai.Client(api_key=Config.GEMINI_API_KEY)
        except Exception as exc:  # pragma: no cover - depend de l'environnement
            _CLIENT_ERROR = str(exc)
            raise LLMUnavailable(f"Client Gemini indisponible : {exc}") from exc
    return _CLIENT


def _log_call(agent: str, duree_ms: int, ok: bool, erreur: str | None = None) -> None:
    with _LOG_LOCK:
        CALL_LOG.append(
            {
                "agent": agent,
                "duree_ms": duree_ms,
                "ok": ok,
                "erreur": erreur,
                "horodatage": time.time(),
            }
        )
        if len(CALL_LOG) > _LOG_MAX:
            del CALL_LOG[: len(CALL_LOG) - _LOG_MAX]


def stats() -> dict:
    """Statistiques d'usage du LLM, affichees dans le dashboard administrateur."""
    with _LOG_LOCK:
        entries = list(CALL_LOG)
    total = len(entries)
    ok = sum(1 for e in entries if e["ok"])
    latences = [e["duree_ms"] for e in entries if e["ok"]]
    return {
        "configure": is_configured(),
        "modele": Config.GEMINI_MODEL,
        "appels_total": total,
        "appels_reussis": ok,
        "appels_echoues": total - ok,
        "latence_moyenne_ms": int(sum(latences) / len(latences)) if latences else 0,
        "taux_succes_pct": round(100 * ok / total, 1) if total else 0.0,
    }


def generate_text(prompt: str, agent: str = "inconnu", temperature: float = 0.4) -> str:
    """Appel texte libre. Leve LLMUnavailable en cas de probleme."""
    client = _get_client()
    debut = time.time()
    try:
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
            config={"temperature": temperature},
        )
        texte = (response.text or "").strip()
        _log_call(agent, int((time.time() - debut) * 1000), True)
        if not texte:
            raise LLMUnavailable("Reponse vide renvoyee par le modele.")
        return texte
    except LLMUnavailable:
        raise
    except Exception as exc:
        _log_call(agent, int((time.time() - debut) * 1000), False, str(exc))
        raise LLMUnavailable(f"Appel a l'API Gemini impossible : {exc}") from exc


def extract_json(texte: str):
    """
    Extrait un objet (ou un tableau) JSON d'une reponse de modele, meme si
    celle-ci est entouree de balises markdown ou de phrases d'introduction.
    """
    texte = texte.strip()
    texte = re.sub(r"^```(?:json)?", "", texte, flags=re.IGNORECASE).strip()
    texte = re.sub(r"```$", "", texte).strip()
    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        pass
    for ouverture, fermeture in (("{", "}"), ("[", "]")):
        debut = texte.find(ouverture)
        fin = texte.rfind(fermeture)
        if debut != -1 and fin > debut:
            fragment = texte[debut : fin + 1]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
    raise ValueError("Aucun JSON valide n'a pu etre extrait de la reponse du modele.")


def generate_json(prompt: str, agent: str = "inconnu", temperature: float = 0.3,
                  tentatives: int = 2):
    """
    Appel attendant une reponse JSON. Reessaie une fois en durcissant la
    consigne si le modele repond hors format.
    """
    dernier_probleme = None
    for essai in range(tentatives):
        consigne = prompt
        if essai > 0:
            consigne = (
                prompt
                + "\n\nATTENTION : ta reponse precedente n'etait pas un JSON valide. "
                "Reponds UNIQUEMENT avec le JSON demande, sans aucun texte autour."
            )
        texte = generate_text(consigne, agent=agent, temperature=temperature)
        try:
            return extract_json(texte)
        except ValueError as exc:
            dernier_probleme = exc
    raise LLMUnavailable(f"Reponse du modele inexploitable : {dernier_probleme}")
