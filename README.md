# EduCompare AI — implémentation complète (9 agents · 6 modules · espace utilisateur)

Application **Flask** implémentant l'architecture décrite au chapitre 3 du rapport de
stage : les **neuf agents d'intelligence artificielle**, les **six modules
fonctionnels**, l'**authentification par compte Google**, la persistance **MongoDB**,
ainsi qu'un **tableau de bord utilisateur** et un **back-office administrateur**.

Un support de cours au format PDF est déposé, analysé par la chaîne complète des neuf
agents, puis comparé à des référentiels pédagogiques étrangers (France, Royaume-Uni,
États-Unis, Canada, Finlande) afin de produire un rapport d'alignement avec des
recommandations priorisées et un export PDF.

---

## 1. Ce qui est implémenté

### Les neuf agents (`app/agents/`)

| # | Agent | Nature | Technologie |
|---|-------|--------|-------------|
| 1 | Extraction | Déterministe | `pypdf` + expressions régulières (repli OCR Tesseract si le PDF est scanné) |
| 2 | Compréhension | Hybride | API Gemini + repli heuristique déterministe |
| 3 | Découpage | Déterministe | Segmentation hiérarchique chapitre → phrases → agrégation avec recouvrement |
| 4 | Vectorisation | **Deep Learning** | `sentence-transformers` multilingue (384 dim.) + repli LSA (TF-IDF + SVD) |
| 5 | Recherche | Déterministe | Base vectorielle **FAISS** `IndexFlatIP` (repli NumPy exact) |
| 6 | Comparaison | **Deep Learning + ML** | Pipeline *Rappel → Précision → Décision* : bi-encodeur, **cross-encodeur** de re-ranking, probabilité de couverture calibrée + lecture qualitative Gemini |
| 7 | Évaluation | **ML + Deep Learning** | Empilement `GradientBoosting` + `MLP` → méta-modèle `Ridge` ; `KMeans`/silhouette ; taxonomie de Bloom ; lisibilité Kandel-Moles |
| 8 | Recommandations | **Apprentissage par renforcement** | MDP + TD(0) sur la valeur d'état, modèle d'acquisition type BKT, graphe de prérequis + rédaction Gemini |
| 9 | Rapport final | Hybride | Agrégation déterministe + synthèse exécutive Gemini |

### Ce que produit l'Agent 6 (au-delà d'un score)

Le bi-encodeur mesure une proximité **thématique**, pas un **enseignement effectif** : un cours qui
évoque les fractions obtient un score élevé face à « conversion fraction / pourcentage » sans jamais
l'enseigner. Un cross-encodeur tranche entre les candidats en traitant la paire conjointement.

Deux questions sont désormais mesurées **séparément** :

| | Question | Sortie |
|---|---|---|
| **Probabilité de couverture** | La notion est-elle abordée ? | 0–100 %, calibrée |
| **Suffisance** | Y a-t-il assez de matière pour qu'elle soit apprise ? | 0–100 % |

D'où une typologie d'écart actionnable — *absente*, *évoquée mais non enseignée*, *amorcée*,
*traitée trop brièvement*, *traitée* — une **zone d'incertitude** explicite (le système dit quand il
ne sait pas, plutôt que de trancher au hasard), et la **preuve textuelle** qui justifie chaque verdict.

### Ce que produit l'Agent 8 (au-delà d'une liste d'écarts)

Un **parcours d'amélioration** ordonné : pour chaque étape, la notion visée, le type d'activité, le
niveau de Bloom ciblé, la maîtrise avant/après prédite, la théorie à ajouter, les exercices gradués,
le critère de réussite, et la **trajectoire de maîtrise** séance après séance.

Le problème est formulé comme un processus de décision markovien :

```
État        maîtrise de chaque notion + prérequis + consensus international
            + gravité de l'écart + budget de séances restant
Action      (notion visée, type d'intervention)   5 types × N notions
Transition  modèle d'acquisition type BKT, amorti si les prérequis manquent
Récompense  gain × importance internationale × gravité − coût
            + prime de palier + récompense terminale de rétention
Politique   a* = argmax [ r(s,a) + γ·V(résumé(s')) ]  — TD(0) model-based
```

**Trois blocs strictement séparés**, plus leur confrontation :

1. `parcours_algorithmique` — nos modèles seuls, aucun texte généré ;
2. `contenu_pedagogique` — Gemini **habille** les étapes déjà décidées (il ne choisit rien) ;
3. `recommandations_gemini` — Gemini seul, libre, clairement étiqueté ;
4. `confrontation` — convergences (diagnostic solide) et divergences (à arbitrer).

**Aucun agent générateur n'est bloquant** : chacun dispose d'un repli local. Si l'API
Gemini est indisponible (quota, réseau, clé absente), l'analyse aboutit quand même et le
rapport signale explicitement quels replis ont été activés, avec un niveau de confiance.

### Les six modules (`app/modules/`)

Les cinq premiers viennent du chapitre 3 du rapport. Ils n'étaient au départ que de
l'orchestration — 2 210 lignes sans un seul modèle. Chacun embarque désormais ses propres
algorithmes, **entraînés sur les données de l'instance, sans aucun modèle de langage**.
Le sixième a été ajouté lorsque le nombre de modèles a rendu leur supervision nécessaire.

| Module | Responsabilité | Modèles | Apport |
|---|---|---|---|
| `module_auth_securite` | Identité, sessions, rôles, habilitations | `IsolationForest` + règles explicites | Score de risque par session, connexions atypiques |
| `module_depot_documents` | Validation et stockage des dépôts | `OneClassSVM` · régression logistique · `MinHash`/LSH | Hors-sujet écartés, matière prédite, doublons détectés |
| `module_traitement_analyse` | Orchestration des 9 agents, suivi, tolérance aux pannes | `GradientBoostingRegressor` · `IsolationForest` | Temps restant réel, exécutions atypiques |
| `module_rapport_restitution` | Contexte du rapport + export PDF | `TextRank` (PageRank) + `MMR` | Synthèse extractive sans modèle de langage |
| `module_historique_dashboard` | Historique, tableaux de bord | `KMeans`/silhouette · `k-NN` · régression | Profils types, trajectoire, analyses similaires |
| `module_supervision_modeles` **(nouveau)** | Dérive des données, état des modèles, réentraînement | `PSI` · Kolmogorov-Smirnov | Détection de la dégradation silencieuse |

#### La contrainte qui a orienté tous les choix

L'instance dispose de très peu de données : 15 analyses, 2 comptes, 34 événements au moment
de la conception. Cela interdit l'apprentissage supervisé sur les données de production et
impose trois principes :

- privilégier les méthodes **non supervisées**, qui n'exigent aucune étiquette ;
- quand une étiquette est indispensable, exploiter le seul corpus réellement étiqueté —
  les 85 notions des référentiels, chacune rattachée à sa matière ;
- pour les modèles qui ont besoin de volume, démarrer sur un jeu **synthétique documenté**,
  puis basculer sur les données réelles dès qu'elles suffisent. Chaque modèle expose son
  état d'amorçage dans le back-office.

#### Ce que ces modèles savent faire, mesuré

| Modèle | Résultat mesuré |
|---|---|
| Triage documentaire | Cours reconnu à **+0,36** ; CV **−1,13**, facture **−1,56**, contrat **−1,32** — 3 négatifs sur 3 écartés |
| Prédiction de matière | **84,6 %** d'exactitude en validation croisée sur 175 textes |
| Détection de doublons | 0,92 sur un quasi-doublon, 0,00 sur un autre chapitre ; LSH partage 32 bandes sur 32 |
| Prédiction de durée | 60,4 s prévues contre 72,9 s réelles, soit **20,6 %** d'écart en amorçage |
| Synthèse extractive | 4 phrases sur 14 (**71 %** de compression), puisées dans 4 chapitres distincts |
| Regroupement des analyses | 3 profils, silhouette **0,66** |
| Dérive (PSI) | 0,025 sur distributions identiques, 2,47 sur une moyenne décalée |

Limite assumée sur la détection d'anomalies de connexion : elle isole bien une connexion
nocturne (risque 86/100) mais **pas** une rafale — `IsolationForest` détecte les points
rares, or une rafale forme un groupe compact. C'est une règle explicite qui la traite.
Environ 5 % des connexions normales sont signalées. Ces chiffres viennent d'un journal
simulé : ils donnent un ordre de grandeur, pas une garantie.

### Espace utilisateur et administration

- **Connexion Google** (OAuth 2.0 / OpenID Connect via Authlib) — aucun mot de passe stocké.
- **Connexion de démonstration** sans Google, pour la soutenance (désactivable).
- **Rôles** : `utilisateur` et `administrateur`.
- **Tableau de bord utilisateur** : note moyenne, progression, activité, historique filtrable.
- **Back-office administrateur** : gestion des comptes (promotion, désactivation), toutes
  les analyses de la plateforme, supervision technique (MongoDB, Gemini, FAISS, modèle
  d'embeddings), journal d'activité, statistiques d'usage.
- **Cloisonnement** : un utilisateur ne peut consulter que ses propres analyses (403 sinon).

### Interface

Design system complet (`app/static/css/design.css`) : thème clair/sombre persistant,
arrière-plan animé, cartes en verre dépoli, jauges circulaires, radar des indicateurs,
anneau de répartition, histogrammes, courbe de progression, écran de suivi temps réel des
neuf agents, navigation mobile dédiée, respect de `prefers-reduced-motion` et feuille d'impression.

Le rapport se lit en six onglets :

| Onglet | Contenu |
|---|---|
| 🎯 **Plan d'amélioration** | Parcours séance par séance, trajectoire de maîtrise, contenu pédagogique déplié (théorie + exercices + critère), grille de suivi de l'élève, contrôle automatique des exercices générés |
| ⚖️ Comparaison | Couverture par référentiel, probabilité et suffisance par notion, preuve textuelle, zone d'incertitude, notions absentes / trop brèves / sans équivalent |
| 📊 Évaluation | 11 indicateurs (radar + barres), modèle d'ensemble et divergence des deux têtes, difficulté et progression, profil de Bloom, profil de maîtrise |
| 🤝 Les deux sources | Colonnes côte à côte — nos modèles vs modèle de langage — puis convergences et divergences |
| 📄 Structure du cours | Chapitres, objectifs, vocabulaire, éléments pédagogiques |
| 🧩 Traçabilité | Chaîne exécutée, benchmark du planificateur, voisinages sémantiques, fiabilité |

L'export PDF (7 sections, ~15 pages) reprend l'intégralité de ces contenus, courbe comprise.

---

## 2. Installation

```bash
cd educompare_prototype
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### 2.1 Configuration

```bash
copy .env.example .env         # Windows
cp .env.example .env           # macOS / Linux
```

Puis renseignez dans `.env` :

| Variable | Rôle |
|----------|------|
| `GEMINI_API_KEY` | Clé gratuite depuis <https://aistudio.google.com/apikey> |
| `GEMINI_MODEL` | Par exemple `gemini-2.0-flash` |
| `FLASK_SECRET_KEY` | Chaîne aléatoire longue (signature des sessions) |
| `MONGO_URI` | `mongodb://localhost:27017` par défaut |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Identifiants OAuth (voir § 2.3) |
| `ADMIN_EMAILS` | Adresses promues automatiquement administrateur |

### 2.2 Modèle sémantique (à faire une fois, avec Internet)

```bash
python setup_download_model.py
```

Télécharge et met en cache `paraphrase-multilingual-MiniLM-L12-v2`. Une fois en cache,
l'agent 4 fonctionne hors ligne. Sans ce modèle, l'agent bascule automatiquement sur le
repli LSA (moins performant en comparaison français/anglais, mais fonctionnel).

### 2.3 Connexion Google (optionnel)

1. Console Google Cloud → *API et services* → *Identifiants* → **ID client OAuth 2.0**
   (type « Application Web »).
2. URI de redirection autorisée : `http://127.0.0.1:5000/connexion/google/callback`
3. Reportez l'ID et le secret dans `.env`.

Sans ces identifiants, la page de connexion propose uniquement le mode démonstration.

### 2.4 MongoDB

Démarrez un serveur MongoDB local (ou pointez `MONGO_URI` vers Atlas). **Si MongoDB est
injoignable, l'application démarre quand même** sur un stockage en mémoire : la
démonstration reste possible, mais les données sont perdues à l'arrêt du processus.
L'état réel est affiché dans le back-office.

### 2.5 PDF de démonstration

```bash
python create_sample_course.py
```

---

## 3. Lancer l'application

```bash
python run.py
```

Puis <http://127.0.0.1:5000>

Parcours : connexion → dépôt du PDF → choix de la matière, du niveau et des pays de
référence → écran de suivi temps réel des 9 agents → rapport interactif → export PDF.

### Test en ligne de commande (sans navigateur)

```bash
python test_pipeline.py sample_data/cours_maths_primaire_exemple.pdf
```

Options : `--matiere`, `--niveau`, `--pays FR UK US`, `--json resultat.json`.

### Tests de non-régression

```bash
pip install -r requirements-dev.txt
pytest             # suite rapide : quelques secondes, hors ligne, sans MongoDB
pytest -m lent     # ancrages : chaîne complète sur le corpus de référence
```

La suite rapide couvre les services transverses, les agents 1 à 3, les règles de
décision des agents 6 et 7, la validation des dépôts et le cloisonnement des
analyses (jusqu'aux codes HTTP). Deux fixtures `autouse` garantissent qu'aucun
test ne peut appeler l'API Gemini ni atteindre un serveur MongoDB.

Les tests d'ancrage exécutent les neuf agents sur dix documents générés
(`tests/corpus_reference/`) et vérifient que note et couverture restent dans les
intervalles **mesurés** de `tests/ancrages.json`. Après un changement de seuil
volontaire, relancer `python tests/mesurer_ancrages.py` et commiter la
différence : le diff devient la trace de l'impact du changement, document par
document.

Détail complet : [`tests/README.md`](tests/README.md).

---

## 4. Structure du code

```
educompare_prototype/
├── run.py                       # point d'entrée Flask
├── test_pipeline.py             # test du pipeline en ligne de commande
├── create_sample_course.py      # génère un PDF de cours d'exemple
├── setup_download_model.py      # téléchargement unique du modèle sémantique
├── requirements.txt · requirements-dev.txt · .env.example · pytest.ini
├── tests/
│   ├── conftest.py              # isolation : base en mémoire, LLM neutralisé
│   ├── test_services.py         # services transverses
│   ├── test_agents.py           # agents 1 à 3, règles de décision 6 et 7
│   ├── test_modules.py          # dépôts, cloisonnement des analyses
│   ├── test_ancrage.py          # chaîne complète (marqué « lent »)
│   ├── mesurer_ancrages.py      # (re)mesure des intervalles de référence
│   ├── ancrages.json            # note et couverture attendues par document
│   └── corpus_reference/        # 10 documents générés + catalogue
└── app/
    ├── __init__.py              # fabrique Flask, OAuth, filtres Jinja, pages d'erreur
    ├── config.py                # configuration centralisée (variables d'environnement)
    ├── agents/                  # les 9 agents (+ CATALOGUE descriptif)
    ├── modules/                 # les 6 modules fonctionnels (+ CATALOGUE_MODULES)
    ├── routes/                  # blueprints : main, auth, dashboard, admin
    ├── models/                  # artefacts entraînés hors ligne (notebooks Colab)
    ├── services/
    │   ├── gemini_client.py     # client LLM partagé (JSON tolérant, journalisation)
    │   ├── database.py          # MongoDB + repli mémoire compatible pymongo
    │   ├── vector_store.py      # FAISS + repli NumPy exact
    │   ├── referentiels.py      # base de connaissances des programmes étrangers (versionnée par pays)
    │   ├── model_registry.py    # registre des 6 modèles + chaîne de replis
    │   ├── reranking.py         # cross-encodeur partagé (agents 6 et 8)
    │   ├── pedagogie.py         # taxonomie de Bloom, lisibilité, progression
    │   ├── prerequis.py         # graphe de prérequis entre notions
    │   ├── rl_parcours.py       # MDP, modèle d'acquisition, politique apprise
    │   ├── entrainement.py      # socle ML des modules : corpus, cache, encodage
    │   ├── empreintes.py        # MinHash + LSH (détection de doublons)
    │   ├── prediction_execution.py  # durée par agent, anomalies d'exécution
    │   ├── synthese_extractive.py   # TextRank + MMR, sans modèle de langage
    │   ├── anomalies_connexion.py   # IsolationForest + règles de sécurité
    │   └── profils_analyses.py  # KMeans, trajectoire, k plus proches voisins
    ├── data/referentiels/       # un dossier par pays, une version figée par fichier
    ├── templates/               # 13 pages Jinja2
    └── static/
        ├── css/design.css       # design system
        └── js/app.js            # interactions (thème, graphiques, suivi temps réel)
```

### Flux d'une analyse

```
Utilisateur → Module Dépôt (validation) → Module Traitement et Analyse
   ↓ fil d'exécution dédié
   Agent 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
   ↓ (l'interface interroge /api/analyse/<id>/progression toutes les 1,2 s)
Module Rapport et Restitution → rapport HTML + export PDF
   ↓
MongoDB (analyse persistée) → Module Historique et Tableau de Bord
```

---

## 5. Comportement en mode dégradé

| Situation | Conséquence |
|-----------|-------------|
| Pas d'Internet / clé Gemini absente ou quota épuisé | Agents 2, 6, 7, 8, 9 basculent sur leur repli. Le parcours, les scores chiffrés, la cartographie et la note restent produits. |
| Modèle `sentence-transformers` non téléchargé | Agent 4 bascule sur LSA (TF-IDF + SVD) ; les seuils de couverture sont recalibrés en conséquence. |
| Cross-encodeur indisponible (`USE_CROSS_ENCODER=false`) | Agent 6 décide sur la similarité cosinus seule ; la probabilité reste calibrée mais moins discriminante. |
| Artefacts `app/models/` absents | Chaque modèle entraîné hors ligne a un repli : voir le registre dans **Administration → Supervision technique**. |
| FAISS non installé | Agent 5 utilise un index NumPy exact — résultats identiques. |
| MongoDB injoignable | Stockage en mémoire, signalé dans le back-office. |
| PDF scanné (aucune couche texte) | OCR tenté si `pytesseract` + `pdf2image` + Tesseract sont installés ; sinon message explicite. |

### Registre des modèles

Six modèles sont déclarés dans `app/services/model_registry.py`, chargés paresseusement et
tous doublés d'un repli. Leur état est visible en temps réel dans le back-office.

| Artefact | Modèle | Agent | Repli si absent |
|---|---|---|---|
| *(HuggingFace)* | Cross-encodeur de re-ranking | 6 | Similarité cosinus seule |
| `couverture_clf.joblib` | Classifieur de couverture calibré | 6 | Fusion logistique documentée |
| `bloom_clf.joblib` | Classification de Bloom | 7 | Heuristique par verbes d'action |
| `niveau_reg.joblib` | Estimateur de niveau scolaire | 7 | Lisibilité Kandel-Moles |
| `dkt_lstm.pt` | Deep Knowledge Tracing | 8 | Modèle d'acquisition type BKT |
| `dqn_planificateur.pt` | Deep Q-Network | 8 | Valeur d'état tabulée → politique gloutonne |

Chaque repli activé est affiché dans le rapport (section « Fiabilité ») et compté dans le
back-office administrateur.

---

## 6. Limites assumées

- **Base de connaissances réduite** : les référentiels étrangers sont un jeu pédagogique
  reconstitué à partir des grandes lignes des programmes officiels, pas leur texte intégral.
  Cette limite est désormais **déclarée dans les données elles-mêmes** (`_meta.nature` à
  `reconstitue`, version `1.0-reconstitue`) et restituée dans le rapport comme dans le
  back-office ; le dépôt d'un texte officiel relu constituera une version `2.0-officiel`.
  Voir [`app/data/referentiels/README.md`](app/data/referentiels/README.md).
- **Modèle d'évaluation (agent 7)** : aucun corpus de cours annotés par des experts n'étant
  disponible à ce stade, l'ensemble est entraîné sur un jeu **synthétique** dérivé de la
  grille d'évaluation experte. Les onze indicateurs, eux, sont bien mesurés sur le document.
  Le modèle est réentraînable tel quel sur des données réelles.
- **Planificateur (agent 8)** : la politique est apprise dans un **simulateur**, pas sur des
  élèves réels. Sur 150 scénarios de test elle dépasse la politique gloutonne sur toutes les
  métriques, mais de peu (+1,8 % de récompense, +2,6 % de rétention, +23 % de consolidation) —
  le chiffre est recalculé à chaque entraînement et affiché dans l'onglet *Traçabilité*.
  Deux formulations antérieures (Q-Learning tabulaire, puis `Q = r + γ·W`) **perdaient** contre
  cette baseline ; l'historique est documenté dans `app/services/rl_parcours.py`.
- **La « maîtrise » est celle attendue d'un élève suivant ce support**, déduite des seules
  caractéristiques du document. Ce n'est en aucun cas la mesure de la maîtrise d'un élève réel.
- **Les exercices générés sont à relire.** Un contrôle automatique de cohérence signale ceux qui
  s'écartent de leur notion cible, mais il ne remplace pas la relecture d'un enseignant.
- **Ollama et n8n** (prévus au chapitre 3) ne sont pas intégrés dans cette version.
- Les scores de similarité constituent une **aide à la décision** et ne remplacent pas
  l'expertise d'un enseignant ou d'un comité d'accréditation.
