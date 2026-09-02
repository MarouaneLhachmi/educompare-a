# EduCompare AI

Application **Flask** qui compare un support de cours de mathématiques aux
programmes scolaires de cinq pays, et en produit un diagnostic d'alignement
assorti d'un parcours d'amélioration.

Un enseignant dépose un document (PDF, Word ou PowerPoint). Neuf agents
d'intelligence artificielle l'analysent en chaîne : extraction, compréhension,
découpage, vectorisation, recherche sémantique, comparaison aux référentiels,
évaluation, recommandations, rapport. Il obtient une couverture notion par
notion — avec la **preuve textuelle** qui justifie chaque verdict — et un
parcours d'amélioration séance par séance.

> **Ce projet est un prototype de démonstration.** Les référentiels étrangers
> sont un jeu **reconstitué** à partir des grandes lignes des programmes
> officiels : ce n'est pas leur texte intégral. Toute conclusion tirée du
> système est bornée par cette approximation, et le système le déclare partout
> où il restitue un résultat.

---

## 1. Périmètre

| | |
|---|---|
| **Matière** | Mathématiques uniquement |
| **Niveaux** | Dernière année du primaire · Dernière année du collège |
| **Référentiels** | 5 pays — France, Royaume-Uni, États-Unis, Canada (Ontario), Finlande |
| **Notions comparées** | 110 au total (55 par niveau) |
| **Version du socle** | `2.0-reconstitue`, versionnée par pays |
| **Formats acceptés** | `.pdf` (avec OCR optionnel), `.docx`, `.pptx` |

Les référentiels sont **versionnés par pays** : chaque analyse enregistre la
signature exacte du socle qui l'a produite (`FR:2.0-reconstitue|UK:…`). Deux
analyses ne sont comparables que si cette signature coïncide — sans quoi une
révision du référentiel se lirait comme une évolution du travail de
l'enseignant. Les versions précédentes ne sont jamais supprimées.

---

## 2. Architecture

Quatre couches, séparation stricte des responsabilités.

```
Présentation      Jinja2 · CSS · JavaScript vanilla   (20 gabarits, 5 blueprints)
Logique métier    Flask · 6 modules fonctionnels      (35 routes)
Intelligence      9 agents · 18 services              (sentence-transformers, FAISS,
                                                       scikit-learn, Gemini)
Persistance       MongoDB + repli mémoire transparent
```

### Les neuf agents

| # | Agent | Nature | Technologie |
|---|---|---|---|
| 1 | Extraction | Déterministe | pypdf · python-docx · python-pptx · repli OCR Tesseract |
| 2 | Compréhension | Hybride | API Gemini + repli heuristique |
| 3 | Découpage | Déterministe | Segmentation hiérarchique avec recouvrement |
| 4 | Vectorisation | Deep Learning | sentence-transformers multilingue (384 dim.) + repli LSA |
| 5 | Recherche | Déterministe | FAISS `IndexFlatIP` + repli NumPy exact |
| 6 | Comparaison | Deep Learning + ML | Bi-encodeur → cross-encodeur → décision calibrée |
| 7 | Évaluation | ML + Deep Learning | GradientBoosting + MLP → Ridge · Bloom · lisibilité |
| 8 | Recommandations | RL + graphe | **Trois sources de priorisation** (voir ci-dessous) |
| 9 | Rapport final | Hybride | Agrégation déterministe + synthèse Gemini |

### Les six modules

`module_auth_securite` · `module_depot_documents` · `module_traitement_analyse`
· `module_rapport_restitution` · `module_historique_dashboard` ·
`module_supervision_modeles`

Chacun embarque ses propres modèles, entraînés sur les données de l'instance,
sans aucun modèle de langage : IsolationForest, OneClassSVM, MinHash/LSH,
TextRank, KMeans, PSI et Kolmogorov-Smirnov.

### Ce qui distingue l'Agent 6

Le bi-encodeur mesure une proximité **thématique**, pas un **enseignement
effectif** : un cours qui évoque les fractions obtient un score élevé face à
« conversion fraction / pourcentage » sans jamais l'enseigner. Un
cross-encodeur tranche ensuite en traitant la paire conjointement.

Deux questions sont donc mesurées **séparément** :

| Indicateur | Question | Sortie |
|---|---|---|
| Probabilité de couverture | La notion est-elle abordée ? | 0–100 %, calibrée |
| Suffisance | Y a-t-il assez de matière pour qu'elle soit apprise ? | 0–100 % |

D'où une typologie d'écart actionnable — absente, évoquée mais non enseignée,
amorcée, traitée trop brièvement, traitée — une **zone d'incertitude
explicite** (le système dit quand il ne sait pas) et la preuve textuelle qui
justifie chaque verdict.

### Ce qui distingue l'Agent 8 : trois sources indépendantes

| Source | Nature | Question posée |
|---|---|---|
| Planificateur RL | MDP + TD(0) sur la valeur d'état | « Quelle action maximise le gain d'apprentissage cumulé ? » |
| Centralité de graphe | PageRank personnalisé, non supervisé | « Quelles notions débloquent le plus d'autres notions ? » |
| Modèle de langage | Gemini, libre et sans contrainte | — |

Les deux algorithmes locaux ne partagent ni leurs entrées décisives ni leur
raisonnement : sur le cours de référence ils ne retiennent que 3 notions
communes sur 19, avec un écart moyen de 3,7 rangs. Une notion retenue par les
**trois** sources constitue un diagnostic particulièrement solide. Quand aucune
ne fait l'unanimité, le rapport le dit — le désaccord est lui-même une
information.

Un quatrième bloc, `contenu_pedagogique`, n'est pas une source : le modèle de
langage y **habille** les étapes déjà décidées par le planificateur, sans rien
choisir.

### Du cours au cursus

Un établissement n'évalue pas un cours mais un **cursus**. L'entité
`Programme` regroupe plusieurs analyses ; la couverture d'une notion devient
alors le **maximum** sur l'ensemble des documents. Une notion absente d'un
chapitre mais traitée dans un autre cesse d'apparaître comme un écart.

La vue programme restitue la matrice notions × documents, l'apport propre de
chaque document (les notions qu'aucun autre ne couvre) et le nombre d'écarts
résorbés par l'agrégation.

---

## 3. Installation

```bash
cd educompare_prototype
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Configuration

```bash
copy .env.example .env         # Windows
cp .env.example .env           # macOS / Linux
```

| Variable | Rôle |
|---|---|
| `GEMINI_API_KEY` | Clé gratuite depuis [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | Par exemple `gemini-2.0-flash` |
| `FLASK_SECRET_KEY` | Chaîne aléatoire longue (signature des sessions) |
| `MONGO_URI` | `mongodb://localhost:27017` par défaut |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Identifiants OAuth (optionnels) |
| `ADMIN_EMAILS` | Adresses promues automatiquement administrateur |

### Modèle sémantique (une fois, avec Internet)

```bash
python setup_download_model.py
```

Met en cache `paraphrase-multilingual-MiniLM-L12-v2`. Une fois téléchargé,
l'Agent 4 fonctionne hors ligne. Sans lui, il bascule sur le repli LSA.

### Lancement

```bash
python run.py
```

Puis <http://127.0.0.1:5000>.

Parcours : connexion → dépôt du document → matière, niveau et pays de
référence → suivi temps réel des 9 agents → rapport interactif → export PDF.

---

## 4. Mode dégradé

Aucun composant génératif n'est bloquant. Chaque brique a son repli, et le
rapport signale explicitement lesquels ont été activés, avec un niveau de
confiance.

| Situation | Conséquence |
|---|---|
| Clé Gemini absente ou quota épuisé | Agents 2, 6, 7, 8 et 9 basculent sur leur repli. Parcours, scores, cartographie et note restent produits. |
| Modèle sentence-transformers absent | Agent 4 bascule sur LSA (TF-IDF + SVD) ; les seuils sont recalibrés. |
| Cross-encodeur indisponible | Agent 6 décide sur la similarité cosinus seule. |
| FAISS non installé | Agent 5 utilise un index NumPy exact — résultats identiques. |
| MongoDB injoignable | Stockage en mémoire, signalé dans le back-office. |
| PDF scanné sans OCR installé | Refusé **au dépôt**, avec un message disant quoi installer, plutôt qu'un échec en cours d'analyse. |

---

## 5. Tests

```bash
pip install -r requirements-dev.txt
pytest                       # 265 tests · ~35 s · hors ligne, sans MongoDB
pytest -m lent               # 35 ancrages + 1 xfail · ~3 min
pytest -m "lent or not lent" # tout
```

**État actuel : 300 tests passent, 1 `xfail` attendu.**

Deux fixtures `autouse` s'appliquent à tous les tests : la base est forcée sur
son repli mémoire, et l'API Gemini est neutralisée. Aucun test ne peut
consommer de quota, dépendre du réseau, ni varier d'une exécution à l'autre.

Les **tests d'ancrage** exécutent la chaîne complète sur 15 documents de
référence et vérifient que note et couverture restent dans les intervalles
*mesurés* de `tests/ancrages.json`. Un ancrage qui échoue ne dit pas « le code
est cassé » mais « un résultat a bougé, justifie-le ».

Détail complet : [`tests/README.md`](tests/README.md).

---

## 6. Structure du code

```
educompare_prototype/
├── run.py · test_pipeline.py · create_sample_course.py · setup_download_model.py
├── requirements.txt · requirements-dev.txt · pytest.ini · .env.example
├── app/
│   ├── agents/          les 9 agents + CATALOGUE descriptif
│   ├── modules/         les 6 modules fonctionnels
│   ├── routes/          main · auth · dashboard · admin · programmes
│   ├── services/        18 services (vectorisation, référentiels, RL, graphe,
│   │                    extraction multi-format, empreintes, pédagogie…)
│   ├── models/          artefacts entraînés hors ligne (les caches sont ignorés par git)
│   ├── data/referentiels/<PAYS>/<version>.json
│   ├── templates/       20 gabarits Jinja2 + _composants.html (macros partagées)
│   └── static/          design.css · app.js
└── tests/               8 fichiers de test + corpus de référence + ancrages
```

---

## 7. Limites connues

### 7.1 Référentiels reconstitués

Les 110 notions sont une paraphrase des grandes lignes des programmes
officiels, pas leur texte intégral. **Toute la chaîne en aval — couverture,
écarts, recommandations, note — est bornée par cette approximation.** Le socle
déclare sa nature (`reconstitue`) dans chaque fichier, et l'annexe
d'accréditation l'affiche en page de garde : un résultat produit sur ce socle
ne peut pas passer pour un relevé officiel.

### 7.2 Détection de chapitres sur documents réels

`_detecter_chapitres` (Agent 1) repère les titres ligne par ligne. Un document
produit par un traitement de texte porte un **en-tête répété à chaque page**,
que l'heuristique compte comme un titre : sur le cours de collège du corpus,
elle détecte **22 chapitres pour un document qui n'en contient qu'un**.

L'impact mesuré reste faible — le découpage produit 33 unités cohérentes et la
couverture n'en est pas notablement affectée — mais la limite touchera tout
document réel. **Correction envisagée** : repérer les lignes identiques d'une
page à l'autre et les écarter avant la détection des titres. Non corrigée à ce
jour.

### 7.3 Portée de la validation du niveau collège

Le niveau collège a été validé avec un cours réel de **1ʳᵉ année**, comparé au
référentiel de **3ᵉ**. Ce test établit que le pipeline traite le collège avec
la même fiabilité et la même profondeur que le primaire — neuf agents, aucun
incident, onze indicateurs, deux parcours de douze étapes, même confiance
déclarée.

Il **n'établit pas** la couverture d'un cours de collège face à son programme :
l'écart mesuré (50,1 % contre 64,9 % au primaire) s'explique par le décalage de
niveau, non par une faiblesse du traitement. Les notions les moins couvertes
sont Pythagore, la trigonométrie et les transformations du plan — réellement
absentes d'un cours de 1ʳᵉ année.

Un vrai test de couverture demanderait soit un cours de 3ᵉ, soit un référentiel
« 1ʳᵉ année de collège » distinct. Ni le référentiel ni le document n'ont été
ajustés pour améliorer le score.

### 7.4 Couverture non monotone de l'Agent 6

Un document contenant 2 chapitres sur 7 obtient **95,3 %** de couverture contre
**64,9 %** pour le document complet dont il est extrait : ajouter du contenu
fait *baisser* la couverture mesurée. Le cross-encodeur note très haut les
paires dont il dispose quand les unités sont peu nombreuses.

Le défaut est encodé en `xfail(strict=True)` : la suite reste verte, le défaut
reste visible, et sa correction fera échouer le marqueur pour qu'on le retire.
Il relève d'une calibration sur étiquettes réelles, pas d'un ajustement de
seuil.

### 7.5 Autres limites assumées

- **Agent 7** : entraîné sur un jeu synthétique dérivé de la grille
  d'évaluation experte, aucun corpus de cours annotés n'étant disponible. Les
  onze indicateurs, eux, sont bien mesurés sur le document.
- **Agent 8** : la politique RL est apprise dans un simulateur, pas sur des
  élèves réels.
- **La « maîtrise »** est celle attendue d'un élève suivant ce support, déduite
  des seules caractéristiques du document. Ce n'est en aucun cas la mesure de
  la maîtrise d'un élève réel.
- **Les exercices générés sont à relire.** Un contrôle automatique de cohérence
  signale ceux qui s'écartent de leur notion cible, sans remplacer la
  relecture d'un enseignant.
- **Prédiction de matière désactivée** : le périmètre ne comptant qu'une seule
  matière, prédire laquelle n'a plus de sens. Le triage documentaire et la
  détection de doublons restent actifs.
- **Ollama et n8n**, prévus au chapitre 3 du rapport de conception, ne sont pas
  intégrés.
- Les scores de similarité constituent une **aide à la décision** et ne
  remplacent ni l'expertise d'un enseignant, ni le jugement d'un comité
  d'accréditation.

---

## 8. Fonctionnalités transverses

- **Authentification** par compte Google (OAuth 2.0 / OpenID Connect) — aucun
  mot de passe stocké. Connexion de démonstration disponible pour la
  soutenance.
- **Cloisonnement** : un enseignant ne consulte que ses propres analyses (403
  sinon), vérifié de bout en bout par les tests.
- **Boucle de retour enseignant** en mode ombre : trois gestes placés là où
  l'enseignant lit déjà, dont les étiquettes sont collectées sans qu'aucun
  comportement du système n'en dépende encore.
- **Annexe d'accréditation** : export austère, sans aucune phrase générée —
  relevé notion par notion avec preuve et page, provenance rejouable, et une
  section entière sur ce que le système **n'a pas** tranché.
- **Back-office** : comptes, analyses de la plateforme, supervision technique
  (MongoDB, Gemini, FAISS, lecteurs de documents, registre des modèles), fil
  d'activité filtrable.
