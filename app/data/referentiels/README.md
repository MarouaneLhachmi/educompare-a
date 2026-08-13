# Référentiels pédagogiques — données versionnées

```
app/data/referentiels/
├── index.json                    liste des pays et de leur version courante
└── <CODE_PAYS>/
    ├── manifeste.json            versions publiées · laquelle est en production
    └── <version>.json            contenu figé
```

## Pourquoi versionner par pays

C'est le programme officiel **d'un pays** qui est révisé, et il l'est pour
toutes ses matières à la fois. Versionner par matière obligerait à republier
cinq fichiers pour une seule réforme et à les garder cohérents entre eux.

Le couple « Matière - Niveau » reste la clé d'accès au contenu, à l'intérieur
de chaque fichier de version.

## Nature d'une version

Le champ `_meta.nature` est le plus important du fichier :

| Valeur | Signification |
|---|---|
| `reconstitue` | Paraphrase des grandes lignes du programme. **Ce n'est pas le texte officiel.** |
| `officiel` | Texte relevé à la source, structuré, puis **relu par un humain** (`relu_par` renseigné). |

Cette distinction est restituée dans le rapport (onglet Traçabilité) et dans
le back-office. Un résultat produit sur des référentiels reconstitués ne doit
pas pouvoir passer pour un résultat produit sur le texte officiel.

**État actuel : les cinq pays sont en `reconstitue`, version `1.0-reconstitue`.**
C'est la limite la plus attaquable du projet, et elle est désormais déclarée
plutôt que tue.

## Déposer une version officielle

1. Relever le texte à la source (voir `_meta.source_officielle` de la version
   courante pour l'adresse de référence de chaque pays).
2. Le structurer en notions `{intitule, descriptif}` sous la bonne clé
   « Matière - Niveau ».
3. Écrire `<CODE_PAYS>/2.0-officiel-<année>.json` avec `_meta.nature` à
   `officiel`, `url_source`, `date_releve` et `relu_par` renseignés.
   **`relu_par` n'est pas décoratif** : une extraction automatique non relue
   reste une version `reconstitue`.
4. Ajouter la version au `manifeste.json` du pays, **sans encore changer**
   `version_courante`.
5. Mesurer l'effet avant de publier :

   ```bash
   pytest -m lent                    # écarts sur le corpus de référence
   python tests/mesurer_ancrages.py  # nouvelle mesure si l'écart est justifié
   ```

6. Basculer `version_courante` dans le manifeste, puis committer le diff de
   `tests/ancrages.json` : il documente l'impact du changement de socle,
   document par document.

Les versions précédentes ne sont jamais supprimées : une analyse ancienne doit
rester interprétable dans les termes du socle qui l'a produite.

## Comparabilité

Chaque analyse enregistre `referentiel_version`, par exemple
`FR:1.0-reconstitue|UK:1.0-reconstitue`. Deux analyses ne sont comparables —
trajectoire, profils, évolution de la note — que si cette signature coïncide.
La trajectoire du tableau de bord applique déjà cette restriction et indique
combien d'analyses ont été écartées.
