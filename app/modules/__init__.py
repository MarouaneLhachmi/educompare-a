"""
Les modules fonctionnels du systeme EduCompare AI.

Les cinq premiers correspondent a la section 3.3.1 du rapport de conception.
Le sixieme a ete ajoute lorsque le nombre de modeles entraines localement a
rendu leur supervision necessaire.

    1. Authentification et Securite      — module_auth_securite
    2. Depot et Gestion des Documents    — module_depot_documents
    3. Traitement et Analyse             — module_traitement_analyse
    4. Rapport et Restitution            — module_rapport_restitution
    5. Historique et Tableau de Bord     — module_historique_dashboard
    6. Supervision des Modeles           — module_supervision_modeles

Chaque module embarque desormais ses propres modeles d'apprentissage,
entraines sur les donnees de l'instance, sans aucun modele de langage.
"""

CATALOGUE_MODULES = [
    {
        "numero": 1,
        "nom": "Authentification et Sécurité",
        "module": "module_auth_securite",
        "role": "Cycle de vie de l'identité : inscription, connexion Google, session, habilitations.",
        "modeles": "IsolationForest sur le comportement de connexion + règles explicites",
        "apport": "Score de risque attribué à chaque session, connexions atypiques signalées.",
        "icone": "🔐",
    },
    {
        "numero": 2,
        "nom": "Dépôt et Gestion des Documents",
        "module": "module_depot_documents",
        "role": "Réception, validation (format, taille, signature) et stockage des supports déposés.",
        "modeles": "One-Class SVM · régression logistique multiclasse · MinHash et LSH",
        "apport": "Documents hors périmètre écartés, matière prédite, quasi-doublons détectés.",
        "icone": "📥",
    },
    {
        "numero": 3,
        "nom": "Traitement et Analyse",
        "module": "module_traitement_analyse",
        "role": "Orchestration séquentielle des neuf agents, suivi d'avancement et tolérance aux pannes.",
        "modeles": "GradientBoostingRegressor pour la durée · IsolationForest sur les profils d'exécution",
        "apport": "Temps restant réellement estimé, exécutions atypiques signalées.",
        "icone": "⚙️",
    },
    {
        "numero": 4,
        "nom": "Rapport et Restitution",
        "module": "module_rapport_restitution",
        "role": "Mise en forme du rapport consultable et génération de l'export PDF.",
        "modeles": "TextRank (PageRank sur graphe de phrases) + MMR",
        "apport": "Synthèse extractive du document, produite sans modèle de langage.",
        "icone": "📊",
    },
    {
        "numero": 5,
        "nom": "Historique et Tableau de Bord",
        "module": "module_historique_dashboard",
        "role": "Vision consolidée des analyses, recherche, filtrage et supervision de la plateforme.",
        "modeles": "KMeans et silhouette · k plus proches voisins · régression linéaire",
        "apport": "Profils types de cours, trajectoire de progression, analyses similaires.",
        "icone": "🗂️",
    },
    {
        "numero": 6,
        "nom": "Supervision des Modèles",
        "module": "module_supervision_modeles",
        "role": "Mesure de la dérive des données, état des modèles et déclenchement du réentraînement.",
        "modeles": "Population Stability Index · test de Kolmogorov-Smirnov",
        "apport": "Détection de la dégradation silencieuse des modèles avant qu'elle ne fausse les analyses.",
        "icone": "🩺",
        "nouveau": True,
    },
]
