"""
Module Depot et Gestion des Documents
======================================

Role (rapport de conception, section 3.3.1) : point d'entree fonctionnel du
processus d'analyse. Ce module prend en charge la reception du document
depose, sa **validation prealable** (format, taille, lisibilite), le
declenchement du traitement, ainsi que la restitution de l'etat d'avancement a
l'utilisateur.

La validation est volontairement effectuee ici, et non dans les routes web :
un document invalide ne doit jamais atteindre le Module Traitement et Analyse.

Triage documentaire par apprentissage
--------------------------------------

La version initiale ne verifiait que la forme : extension, taille, signature
PDF. Un curriculum vitae renomme en `.pdf` la franchissait sans difficulte et
declenchait cent secondes de pipeline pour un resultat vide de sens.

Le module realise desormais un **controle avant vol** : une pre-extraction
legere des premieres pages, puis trois modeles complementaires, tous
entraines localement et sans aucun modele de langage.

1. **Est-ce un support de cours ?** — `OneClassSVM` sur les vecteurs
   semantiques. L'apprentissage a une classe est le choix impose par les
   donnees : on dispose d'exemples de contenus pedagogiques (les notions des
   referentiels, les unites des analyses passees) mais d'aucun exemple de ce
   qu'il faut rejeter. Un classifieur binaire exigerait des contre-exemples
   qu'il faudrait inventer ; un modele a une classe apprend la frontiere du
   normal et signale ce qui en sort.

2. **De quelle matiere s'agit-il ?** — `LogisticRegression` multiclasse,
   entrainee sur le seul corpus reellement etiquete de l'instance : les
   notions des referentiels, chacune rattachee a sa matiere. Le module
   propose la matiere au lieu de la demander, et signale un desaccord avec
   la saisie de l'utilisateur.

3. **Ce document a-t-il deja ete analyse ?** — empreinte `MinHash` et
   filtrage `LSH` (voir `services/empreintes`), qui estiment la similarite de
   Jaccard entre documents a cout constant.

Aucun de ces trois modeles n'est bloquant : leur indisponibilite ramene le
module a sa validation de forme d'origine.
"""

import os
import re
import unicodedata
import uuid
from datetime import datetime

import numpy as np
from werkzeug.utils import secure_filename

from app.config import Config
from app.services import empreintes, entrainement, extraction_documents


class DocumentInvalide(Exception):
    """Levee lorsqu'un document ne passe pas la validation prealable."""


TAILLE_MIN_OCTETS = 1024  # 1 Ko : en dessous, le fichier ne peut pas etre un cours

# --- Parametres du triage --------------------------------------------------
# Pages lues lors de la pre-extraction : assez pour caracteriser le document,
# assez peu pour rester instantane.
PAGES_PRE_EXTRACTION = 4
# En dessous de ce volume de texte, aucun modele ne peut se prononcer.
MOTS_MINIMUM_TRIAGE = 40
# Part de contenu pedagogique attendue en marge de la frontiere apprise :
# 0,12 tolere que 12 % du corpus d'entrainement soit considere atypique, ce
# qui evite un modele trop etroit sur un corpus encore reduit.
NU_ONE_CLASS = 0.12
# Seuil de confiance en dessous duquel la matiere predite n'est pas proposee.
CONFIANCE_MATIERE_MIN = 0.45
# Similarite de Jaccard au-dela de laquelle deux documents sont consideres
# comme un quasi-doublon.
SEUIL_DOUBLON = 0.55

# Signature binaire attendue par format. Les formats bureautiques modernes
# sont des archives ZIP : « PK\x03\x04 » ne distingue donc pas un .docx d'un
# .pptx, et ce n'est pas son role — il s'agit d'ecarter un fichier dont le
# contenu n'a rien a voir avec son extension.
SIGNATURES = {
    "pdf": b"%PDF",
    "docx": b"PK\x03\x04",
    "pptx": b"PK\x03\x04",
}


def _extension(nom_fichier: str) -> str:
    return nom_fichier.rsplit(".", 1)[-1].lower() if "." in nom_fichier else ""


def _nom_sur(nom_fichier: str) -> str:
    """
    `secure_filename` supprime les caracteres accentues : on translittere
    d'abord pour conserver un nom lisible (« Cours de mathématiques.pdf »
    devient « Cours_de_mathematiques.pdf » et non « Cours_de_mathmatiques.pdf »).
    """
    normalise = unicodedata.normalize("NFKD", nom_fichier)
    sans_accents = "".join(c for c in normalise if not unicodedata.combining(c))
    nettoye = secure_filename(sans_accents)
    return nettoye or f"document_{uuid.uuid4().hex[:6]}.pdf"


def valider(fichier) -> None:
    """
    Verifie le format et la taille du fichier recu.
    Leve `DocumentInvalide` avec un message affichable a l'utilisateur.
    """
    if fichier is None or not getattr(fichier, "filename", ""):
        raise DocumentInvalide("Aucun fichier n'a été déposé.")

    extension = _extension(fichier.filename)
    if extension not in Config.ALLOWED_EXTENSIONS:
        autorisees = ", ".join(sorted(Config.ALLOWED_EXTENSIONS)).upper()
        raise DocumentInvalide(
            f"Format « .{extension or '?'} » non pris en charge. Formats acceptés : {autorisees}."
        )

    # Taille : le flux est repositionne apres mesure pour rester lisible.
    fichier.stream.seek(0, os.SEEK_END)
    taille = fichier.stream.tell()
    fichier.stream.seek(0)

    if taille < TAILLE_MIN_OCTETS:
        raise DocumentInvalide("Le fichier déposé est vide ou trop petit pour être un support de cours.")
    if taille > Config.MAX_CONTENT_LENGTH:
        limite_mo = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
        raise DocumentInvalide(f"Le fichier dépasse la taille maximale autorisée ({limite_mo} Mo).")

    # Signature du fichier : evite qu'un document renomme traverse le pipeline.
    # Un .docx comme un .pptx sont des archives ZIP, d'ou une signature
    # commune ; c'est le lecteur du format qui tranchera ensuite.
    entete = fichier.stream.read(5)
    fichier.stream.seek(0)
    signature_attendue = SIGNATURES.get(extension)
    if signature_attendue and not entete.startswith(signature_attendue):
        raise DocumentInvalide(
            f"Le contenu du fichier ne correspond pas à un {extension.upper()} "
            f"valide (signature absente)."
        )


def enregistrer(fichier, utilisateur: dict | None = None) -> dict:
    """
    Valide puis stocke le document sur le disque, sous un nom unique afin que
    deux depots successifs du meme fichier ne s'ecrasent pas.

    Retourne les metadonnees du document (equivalent de la classe `Document`
    du diagramme de classes).
    """
    valider(fichier)

    nom_original = fichier.filename
    nom_sur = _nom_sur(nom_original)
    racine, extension = os.path.splitext(nom_sur)
    identifiant = uuid.uuid4().hex[:8]
    nom_stocke = f"{racine[:60]}_{identifiant}{extension or '.pdf'}"

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    chemin = os.path.join(Config.UPLOAD_FOLDER, nom_stocke)
    fichier.save(chemin)

    taille = os.path.getsize(chemin)
    return {
        "id": identifiant,
        "nom_original": nom_original,
        "nom_stocke": nom_stocke,
        "chemin": chemin,
        "taille_octets": taille,
        "taille_lisible": taille_lisible(taille),
        "date_depot": datetime.now().isoformat(),
        "depose_par": (utilisateur or {}).get("id"),
    }


def enregistrer_avec_triage(fichier, utilisateur: dict | None = None,
                            matiere_declaree: str | None = None) -> dict:
    """
    Stocke le document puis le soumet au triage.

    L'empreinte MinHash est conservee avec les metadonnees du document :
    c'est elle qui permettra de reconnaitre un depot ulterieur du meme
    support. Le triage ne bloque rien ; ses alertes sont remontees a
    l'interface, qui decide de leur presentation.
    """
    document = enregistrer(fichier, utilisateur)

    try:
        diagnostic = analyser_contenu(
            document["chemin"], matiere_declaree, (utilisateur or {}).get("id")
        )
    except Exception as exc:
        # Le triage est un confort, jamais une condition : son echec ne doit
        # pas empecher le depot.
        document["triage"] = {"applique": False, "erreur": str(exc)[:200]}
        return document

    document["empreinte"] = diagnostic.pop("empreinte", None)
    document["seaux"] = diagnostic.pop("seaux", None)
    document["triage"] = diagnostic
    return document


def taille_lisible(octets: int) -> str:
    for unite in ("o", "Ko", "Mo", "Go"):
        if octets < 1024 or unite == "Go":
            return f"{octets:.0f} {unite}" if unite == "o" else f"{octets:.1f} {unite}"
        octets /= 1024
    return f"{octets:.1f} Go"


def supprimer_fichier(chemin: str) -> bool:
    """Supprime le fichier physique associe a une analyse (purge back-office)."""
    try:
        if chemin and os.path.exists(chemin) and os.path.commonpath(
            [os.path.abspath(chemin), os.path.abspath(Config.UPLOAD_FOLDER)]
        ) == os.path.abspath(Config.UPLOAD_FOLDER):
            os.remove(chemin)
            return True
    except Exception:
        pass
    return False


def statistiques_stockage() -> dict:
    """Occupation disque des dossiers de depot et de sortie (supervision)."""
    def mesurer(dossier: str) -> tuple[int, int]:
        if not os.path.isdir(dossier):
            return 0, 0
        fichiers = [
            os.path.join(dossier, f)
            for f in os.listdir(dossier)
            if os.path.isfile(os.path.join(dossier, f))
        ]
        return len(fichiers), sum(os.path.getsize(f) for f in fichiers)

    nb_depots, poids_depots = mesurer(Config.UPLOAD_FOLDER)
    nb_rapports, poids_rapports = mesurer(Config.OUTPUT_FOLDER)
    return {
        "documents_deposes": nb_depots,
        "poids_depots": taille_lisible(poids_depots),
        "rapports_generes": nb_rapports,
        "poids_rapports": taille_lisible(poids_rapports),
        "limite_par_fichier": f"{Config.MAX_CONTENT_LENGTH // (1024 * 1024)} Mo",
    }


_MOTIF_ID = re.compile(r"^[a-f0-9]{6,32}$")


def identifiant_valide(valeur: str) -> bool:
    """Garde-fou contre les identifiants forges dans les URL."""
    return bool(valeur and _MOTIF_ID.match(valeur))


# ===========================================================================
# Triage documentaire
# ===========================================================================

def pre_extraire(chemin: str, max_pages: int = PAGES_PRE_EXTRACTION) -> dict:
    """
    Lecture legere des premieres pages, avant toute analyse.

    Ne remplace pas l'Agent 1 : il s'agit d'obtenir juste assez de texte pour
    que les modeles de triage se prononcent, en quelques dizaines de
    millisecondes.
    """
    format_source = extraction_documents.format_de(chemin)
    try:
        # `tenter_ocr=False` : la pre-extraction doit rester instantanee. Un
        # PDF scanne ressort donc illisible ici, et c'est le diagnostic qui
        # proposera l'OCR plutot que de refuser le document.
        lecture = extraction_documents.lire(chemin, max_pages=max_pages,
                                            tenter_ocr=False)
        texte = "\n".join(lecture["pages"])
        nb_pages = lecture["nb_pages_document"]
    except extraction_documents.DocumentIllisible as exc:
        return {"texte": "", "nb_pages": 0, "nb_mots": 0, "lisible": False,
                "format": format_source, "erreur": str(exc)[:200]}
    except Exception as exc:
        return {"texte": "", "nb_pages": 0, "nb_mots": 0, "lisible": False,
                "format": format_source, "erreur": str(exc)[:160]}

    mots = texte.split()
    caracteres = len(texte)
    alphabetiques = sum(1 for c in texte if c.isalpha())
    pages_lues = max(1, min(nb_pages, max_pages))

    return {
        "texte": texte,
        "nb_pages": nb_pages,
        "format": format_source,
        "nb_mots": len(mots),
        "caracteres_par_page": round(caracteres / pages_lues, 1),
        # Un PDF scanne ou corrompu produit surtout des caracteres de
        # controle : la part d'alphabetiques distingue un texte exploitable.
        "part_alphabetique": round(alphabetiques / caracteres, 3) if caracteres else 0.0,
        "lisible": len(mots) >= MOTS_MINIMUM_TRIAGE,
        "erreur": None,
    }


def _corpus_pedagogique() -> tuple[list[str], list[str]]:
    """
    Corpus d'apprentissage du triage : notions des referentiels et unites de
    contenu des analyses deja realisees.

    Les notions apportent la couverture thematique, les unites apportent le
    registre reel d'un support de cours. Le corpus s'enrichit donc a chaque
    analyse, sans intervention.

    **Le perimetre fait autorite.** Seules les analyses portant une matiere
    encore presente dans les referentiels alimentent le corpus. Les analyses
    des matieres retirees restent en base — elles sont conservees pour la
    tracabilite, et supprimer un historique pour faire plaisir a un modele
    serait une mauvaise facon de regler le probleme — mais elles ne doivent
    plus apprendre au classifieur a predire une matiere que le systeme ne sait
    plus comparer. Le filtre est deduit des referentiels, pas ecrit en dur :
    il suit automatiquement le prochain changement de perimetre.
    """
    reference = entrainement.corpus_referentiels()
    textes = list(reference["textes"])
    matieres = list(reference["matieres"])
    perimetre = set(reference["matieres_distinctes"])

    for analyse in entrainement.corpus_analyses():
        matiere = analyse.get("matiere")
        if matiere not in perimetre:
            continue
        for unite in ((analyse.get("agent3") or {}).get("unites") or [])[:12]:
            texte = (unite.get("texte") or "").strip()
            if len(texte.split()) >= 15:
                textes.append(texte)
                matieres.append(matiere)
    return textes, matieres


def _entrainer_triage() -> dict:
    """Frontiere du « contenu pedagogique » apprise a une seule classe."""
    from sklearn.svm import OneClassSVM

    textes, _ = _corpus_pedagogique()
    vecteurs, source = entrainement.encoder_textes(textes)
    if vecteurs is None or len(vecteurs) < 20:
        return {"modele": None, "source_donnees": source, "nb_observations": len(textes)}

    modele = OneClassSVM(kernel="rbf", gamma="scale", nu=NU_ONE_CLASS)
    modele.fit(vecteurs)

    # Distribution des scores sur le corpus d'entrainement : elle sert a
    # normaliser le score d'un nouveau document sur une echelle lisible.
    scores = modele.decision_function(vecteurs)
    return {
        "modele": modele,
        "source_donnees": f"{source} — notions de référentiels + unités des analyses passées",
        "nb_observations": len(textes),
        "metrique_score_median": round(float(np.median(scores)), 4),
        "metrique_score_p05": round(float(np.percentile(scores, 5)), 4),
        "metrique_part_atypique_apprentissage": round(float(np.mean(scores < 0)), 3),
    }


def _entrainer_matiere() -> dict:
    """Classifieur de matiere, entraine sur les notions etiquetees."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    textes, matieres = _corpus_pedagogique()
    if len(set(matieres)) < 2:
        return {"modele": None, "source_donnees": "corpus mono-matière",
                "nb_observations": len(textes)}

    vecteurs, source = entrainement.encoder_textes(textes)
    if vecteurs is None:
        return {"modele": None, "source_donnees": source, "nb_observations": len(textes)}

    modele = LogisticRegression(max_iter=1500, C=2.0, class_weight="balanced")
    modele.fit(vecteurs, matieres)

    # Validation croisee : sur un corpus de cette taille, c'est la seule
    # mesure honnete de ce que vaut le classifieur.
    try:
        scores = cross_val_score(
            LogisticRegression(max_iter=1500, C=2.0, class_weight="balanced"),
            vecteurs, matieres, cv=min(5, min(np.bincount(
                np.unique(matieres, return_inverse=True)[1]))),
        )
        exactitude = round(float(np.mean(scores)), 3)
    except Exception:
        exactitude = None

    return {
        "modele": modele,
        "source_donnees": f"{source} — notions de référentiels étiquetées par matière",
        "nb_observations": len(textes),
        "metrique_exactitude_validation_croisee": exactitude,
        "metrique_classes": sorted(set(matieres)),
    }


def analyser_contenu(chemin: str, matiere_declaree: str | None = None,
                     utilisateur_id: str | None = None) -> dict:
    """
    Triage complet d'un document depose : nature, matiere, doublon.

    Retourne un diagnostic exploitable par l'interface. Aucun rejet n'est
    prononce ici : le module signale, l'utilisateur decide. Un modele de
    triage entraine sur un corpus encore reduit ne doit pas avoir le dernier
    mot sur le travail d'un enseignant.
    """
    apercu = pre_extraire(chemin)
    diagnostic = {
        "apercu": {k: v for k, v in apercu.items() if k != "texte"},
        "extrait": apercu["texte"][:400],
        "triage": {"applique": False},
        "matiere": {"applique": False},
        "doublon": {"applique": False},
        "alertes": [],
    }

    if not apercu["lisible"]:
        # Document sans couche texte : plutot que de refuser, on dit ce qui
        # est possible. Si l'OCR est installe, l'analyse peut aboutir — le
        # controle au depot ne le tente pas lui-meme pour rester instantane.
        ocr = extraction_documents.dependances()["ocr"]
        scanne = apercu["nb_mots"] == 0 and apercu.get("format") == "pdf"

        if scanne and ocr["disponible"]:
            message = (
                "Aucune couche texte n'a été trouvée : le document est probablement "
                "scanné. La reconnaissance optique de caractères est installée et "
                "sera appliquée pendant l'analyse — comptez un traitement plus long."
            )
            niveau = "alerte"
        elif scanne:
            message = (
                "Aucune couche texte n'a été trouvée : le document est probablement "
                "scanné, et la reconnaissance optique de caractères n'est pas "
                "installée sur ce serveur (pytesseract, pdf2image et Tesseract). "
                "L'analyse échouera en l'état."
            )
            niveau = "erreur"
        elif apercu["nb_mots"] == 0:
            message = (
                f"Aucun texte n'a pu être lu dans ce document "
                f"({extraction_documents.LIBELLES_FORMAT.get(apercu.get('format'), 'format inconnu')}). "
                + (apercu.get("erreur") or "Le document est peut-être vide.")
            )
            niveau = "erreur"
        else:
            message = (
                f"Très peu de texte détecté ({apercu['nb_mots']} mots sur les "
                f"premières pages) : l'analyse risque d'être peu significative."
            )
            niveau = "alerte"

        diagnostic["alertes"].append({"niveau": niveau, "message": message})
        diagnostic["ocr"] = {
            "propose": scanne,
            "disponible": ocr["disponible"],
            "detail": ocr,
        }
        return diagnostic

    vecteurs, _ = entrainement.encoder_textes([apercu["texte"][:4000]])
    if vecteurs is None:
        return diagnostic

    # --- 1. Nature du document ------------------------------------------
    triage = entrainement.obtenir("depot_triage", _entrainer_triage)
    if triage.get("modele") is not None:
        score = float(triage["modele"].decision_function(vecteurs)[0])
        reference = abs(triage.get("metrique_score_p05") or 1.0) or 1.0
        # Score normalise : 1 = au coeur du corpus pedagogique, 0 = a la
        # frontiere, negatif = hors distribution.
        confiance = float(np.clip(0.5 + score / (2 * reference), 0.0, 1.0))
        est_pedagogique = score >= 0
        diagnostic["triage"] = {
            "applique": True,
            "est_support_de_cours": est_pedagogique,
            "score": round(score, 4),
            "confiance": round(confiance, 3),
            "modele": "OneClassSVM (noyau RBF)",
            "nb_observations": triage["nb_observations"],
            "amorcage": triage.get("amorcage", True),
        }
        if not est_pedagogique:
            diagnostic["alertes"].append({
                "niveau": "alerte",
                "message": (
                    "Ce document ne ressemble pas aux supports de cours connus du système. "
                    "L'analyse reste possible, mais ses résultats risquent d'être peu "
                    "pertinents. Vérifiez qu'il s'agit bien d'un support pédagogique."
                ),
            })

    # --- 2. Matiere -------------------------------------------------------
    classifieur = entrainement.obtenir("depot_matiere", _entrainer_matiere)
    if classifieur.get("modele") is not None:
        modele = classifieur["modele"]
        probabilites = modele.predict_proba(vecteurs)[0]
        ordre = np.argsort(-probabilites)
        predite = str(modele.classes_[ordre[0]])
        confiance = float(probabilites[ordre[0]])
        diagnostic["matiere"] = {
            "applique": True,
            "predite": predite,
            "confiance": round(confiance, 3),
            "fiable": confiance >= CONFIANCE_MATIERE_MIN,
            "classement": [
                {"matiere": str(modele.classes_[i]), "probabilite": round(float(probabilites[i]), 3)}
                for i in ordre[:3]
            ],
            "modele": "Régression logistique multiclasse",
            "exactitude_validation": classifieur.get("metrique_exactitude_validation_croisee"),
        }
        # On ne signale un desaccord de matiere que si le document a par
        # ailleurs ete reconnu comme un support de cours : sur un document
        # hors perimetre, le classifieur est contraint de choisir une classe
        # et sa prediction n'a aucun sens.
        pertinent = diagnostic["triage"].get("est_support_de_cours", True)
        if (pertinent and matiere_declaree and confiance >= CONFIANCE_MATIERE_MIN
                and predite != matiere_declaree):
            diagnostic["alertes"].append({
                "niveau": "alerte",
                "message": (
                    f"La matière déclarée est « {matiere_declaree} » mais le contenu "
                    f"ressemble davantage à « {predite} » ({confiance:.0%} de confiance). "
                    f"Une matière erronée fausse toute la comparaison."
                ),
            })

    # --- 3. Doublon --------------------------------------------------------
    signature = empreintes.empreinte(apercu["texte"])
    diagnostic["empreinte"] = signature
    diagnostic["seaux"] = empreintes.seaux_lsh(signature)

    candidats = []
    for analyse in entrainement.corpus_analyses():
        if utilisateur_id and analyse.get("utilisateur_id") != utilisateur_id:
            continue
        document = analyse.get("document") or {}
        if document.get("empreinte"):
            candidats.append({
                "analyse_id": analyse.get("id"),
                "nom_fichier": analyse.get("nom_fichier"),
                "date": analyse.get("date_creation"),
                "empreinte": document["empreinte"],
                "seaux": document.get("seaux"),
            })

    proches = empreintes.chercher_proches(signature, candidats, seuil=SEUIL_DOUBLON)
    diagnostic["doublon"] = {
        "applique": True,
        "nb_candidats_compares": len(candidats),
        "proches": [
            {k: v for k, v in p.items() if k not in ("empreinte", "seaux")}
            for p in proches
        ],
        **empreintes.infos(),
    }
    if proches:
        plus_proche = proches[0]
        diagnostic["alertes"].append({
            "niveau": "info",
            "message": (
                f"Ce document ressemble à {plus_proche['similarite']:.0%} à « "
                f"{plus_proche['nom_fichier']} », analysé le {plus_proche['date']}. "
                f"Une nouvelle analyse produira des résultats très proches."
            ),
        })

    return diagnostic


def infos_modeles_triage() -> list[dict]:
    """Etat des modeles de triage, pour la supervision technique."""
    return [
        etat for etat in entrainement.etat_modeles()
        if etat["nom"].startswith("depot_")
    ]
