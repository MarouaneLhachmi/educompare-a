# Tests de non-régression — EduCompare AI

Première étape du plan de transition (phase 1, point 1.3). Ce qui manquait
jusqu'ici n'était pas la vérification — elle était faite, manuellement, à
chaque étape du développement — mais sa **trace exécutable** : rien ne rejouait
ces vérifications automatiquement au changement suivant.

## Lancer les tests

```bash
pytest
```

194 tests — services, agents 1 à 3, règles de décision, modules, lecture
multi-format, programmes, accès HTTP. **Environ 5 secondes**, hors ligne, sans MongoDB. Aucun test de cette suite ne
charge le modèle sémantique : c'est ce qui la rend utilisable à chaque
enregistrement de fichier.

```bash
pytest -m lent
```

33 tests d'ancrage : chaîne complète des neuf agents sur les douze documents
du corpus de référence. Environ 2 min 40 s, dominées par le pipeline lui-même.

```bash
pytest -m "lent or not lent"
```

Tout.

## Ce que garantit le socle

Deux fixtures `autouse` de `conftest.py` s'appliquent à **tous** les tests,
sans qu'aucun ait à y penser :

| Fixture | Effet |
|---|---|
| `base_memoire` | `services.database` est forcé sur son repli en mémoire, vidé avant chaque test. Aucun serveur MongoDB n'est requis, aucune donnée ne survit à un test. |
| `llm_hors_ligne` | `generate_text` et `generate_json` lèvent `LLMUnavailable`. Aucun test ne peut appeler l'API Gemini — ni consommer de quota, ni dépendre du réseau, ni varier d'une exécution à l'autre. |

Les agents génératifs empruntent donc systématiquement leur repli local. Ce
n'est pas un contournement : c'est le mode dégradé que le système revendique,
et la propriété la plus importante à protéger. Un test qui veut vérifier le
chemin nominal demande explicitement la fixture `llm_simule`, qui installe un
faux client auquel on dépose les réponses attendues.

## Le corpus de référence

`corpus_reference/` contient douze documents **générés** par
`generer_corpus.py` — donc reproductibles à l'identique sur n'importe quelle
machine, sans question de droit d'auteur ni de donnée personnelle.

| Document | Nature | Ce qu'il teste |
|---|---|---|
| `cours_maths_complet.pdf` | cours complet, 7 chapitres | référence haute de couverture |
| `cours_maths_partiel.pdf` | 2 chapitres sur 7 | la couverture doit chuter |
| `cours_maths_doublon.pdf` | quasi-doublon du complet | détection MinHash/LSH, stabilité du diagnostic |
| `plan_de_cours.pdf` | intitulés sans contenu enseigné | notions *évoquées* mais non *enseignées* |
| `cours_sciences.pdf` | cours hors périmètre | un vrai support pédagogique que le système ne sait plus comparer — depuis le recentrage sur les mathématiques, sa couverture doit s'effondrer sans que rien ne plante |
| `cours_maths_anglais.pdf` | cours en anglais | comparaison multilingue (dégradée par le repli LSA) |
| `hors_sujet_cv.pdf` | CV | triage documentaire |
| `hors_sujet_facture.pdf` | facture | triage documentaire |
| `hors_sujet_contrat.pdf` | contrat | triage documentaire |
| `cours_maths.docx` | document Word | styles de titre, contenu des tableaux |
| `cours_maths.pptx` | présentation | une page par diapositive, notes du présentateur |
| `scan_sans_texte.pdf` | aucune couche texte | échec explicite de l'Agent 1, alerte au dépôt |

Les deux documents bureautiques nomment leurs sections **sans le mot
« Chapitre »** : l'heuristique du PDF les manquerait. Ils vérifient que la
structure déclarée par l'auteur — styles de titre, titres de diapositive — est
bien préférée à la structure devinée.

Régénérer après modification :

```bash
python tests/corpus_reference/generer_corpus.py
```

## Les ancrages

`ancrages.json` contient, pour chaque document, la note et la couverture
**mesurées** sur le comportement actuel du système, entourées d'une marge de
tolérance (±8 points sur la note, ±10 sur la couverture). Les tests d'ancrage
vérifient que le système reste dans cette bande.

Un ancrage n'est pas une valeur décrétée. Le fichier enregistre aussi
l'environnement de la mesure — moteur de vectorisation, cross-encodeur actif,
seuils de l'Agent 6 — parce qu'un intervalle mesuré avec le modèle neuronal
n'a pas de sens sur une machine tombée en repli LSA.

Regénérer :

```bash
python tests/mesurer_ancrages.py
```

### Quand un ancrage échoue

**Ça ne veut pas dire que le code est cassé. Ça veut dire qu'un résultat a
bougé, et qu'il faut le justifier.**

C'est exactement l'outil qui manquait lors du durcissement de l'Agent 6 : la
note du cours de démonstration est passée de 74,9 à 54. C'était le résultat
recherché — un diagnostic plus honnête — mais rien dans le système ne l'aurait
signalé dans le cas contraire.

Marche à suivre :

1. Le déplacement est-il **voulu** ? Si oui, relancer `mesurer_ancrages.py` et
   **commiter la différence** : le diff de `ancrages.json` devient la trace
   explicite de l'impact du changement, document par document.
2. Sinon, `pytest tests/test_agents.py -k Decision` désigne généralement la
   cause : ce sont les fonctions qui traduisent un score en verdict.
3. Vérifier enfin `environnement` dans `ancrages.json` : un repli de
   vectorisation actif suffit à déplacer tous les scores sans qu'aucune ligne
   de code ait changé.

## Ce que la première mesure a révélé

Le corpus a trouvé un défaut dès sa première exécution, avant même d'avoir
servi à protéger quoi que ce soit.

`cours_maths_partiel.pdf` contient **les deux premiers chapitres** de
`cours_maths_complet.pdf`, et rien d'autre. Il obtient **95,3 % de couverture,
contre 64,9 % pour le document complet** : ajouter cinq chapitres fait *baisser*
la couverture mesurée. Une mesure de couverture ne peut pas être non monotone
sur un sur-ensemble de contenu.

Mécanisme, vérifié pair par pair : avec seulement deux unités de sens, le
cross-encodeur note très haut les paires dont il dispose. Pour la notion
« Solides et volumes », il attribue un logit de **+3,2 au chapitre sur les
fractions** et de **−2,3 au chapitre de géométrie**, qui est pourtant celui qui
traite cube, pavé droit et cylindre. Sur le document complet, le bi-encodeur
propose d'autres candidats et l'effet est masqué : le bon résultat y est obtenu
pour la mauvaise raison. Sur les 385 paires (7 unités × 55 notions) du cours
complet, la corrélation de Spearman entre similarité cosinus et logit du
cross-encodeur n'est que de **0,11**, et les deux moteurs élisent la même unité
pour 17 notions sur 55 (le hasard en donnerait 8).

Le défaut est encodé dans `test_ancrage.py` en `xfail(strict=True)` : la suite
reste verte, le défaut reste visible, et le jour où il sera corrigé le marqueur
provoquera un échec demandant son retrait. Un second test surveille que
l'inversion ne s'aggrave pas.

Sa correction ne relève pas d'un ajustement de seuil — elle relève du point 1.2
du plan de transition, la bascule de `couverture_clf` sur des étiquettes réelles
issues du retour enseignant. C'est aussi une illustration directe de l'argument
du plan : les seuils actuels sont réglés à la main, et rien ne les valide contre
un jugement humain.

## Organisation

| Fichier | Contenu |
|---|---|
| `conftest.py` | isolation (base, LLM), corpus, entrées figées, client Flask |
| `test_services.py` | base en mémoire, référentiels, extraction JSON, MinHash, Bloom, synthèse extractive (repli lexical ; le chemin neuronal est marqué `lent`) |
| `test_extraction_documents.py` | lecture PDF / Word / PowerPoint, orientation vers l'OCR |
| `test_programmes.py` | agrégation d'un cursus, comparabilité, routes des programmes |
| `test_agents.py` | agents 1 à 3 sur entrées figées ; règles de décision des agents 6 et 7 |
| `test_modules.py` | validation des dépôts, pré-extraction, cloisonnement des analyses (unitaire **et** HTTP) |
| `test_ancrage.py` | chaîne complète : intervalles, invariants pédagogiques, mode dégradé, restitution |
| `mesurer_ancrages.py` | (re)mesure des intervalles |
| `corpus_reference/generer_corpus.py` | génération des dix documents |

## Conventions

Les tests visent des **propriétés revendiquées par le code**, pas des détails
d'implémentation. Quand un test protège une décision de conception documentée
dans les commentaires du code, il le dit dans sa docstring et cite le cas
concret — par exemple le faux positif « la formule » → verbe « formuler », qui
ferait basculer un simple calcul au niveau « Créer » de la taxonomie de Bloom.

Un test dont on ne peut pas écrire pourquoi son échec serait grave n'a pas sa
place ici.
