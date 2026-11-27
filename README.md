# Initiation au RAG : six retrievers LlamaIndex avec Chroma DB

Ce projet est une initiation au **RAG** (*Retrieval-Augmented Generation*, ou génération augmentée par la recherche).
Au lieu de répondre uniquement avec ses connaissances, le modèle de langage reçoit d'abord les passages les plus
utiles trouvés dans vos documents, puis rédige sa réponse à partir de ces passages.

Ici, les documents sont des **pages web scrapées**. Le projet permet d'essayer **six retrievers** (stratégies de
recherche) sur les mêmes données et de les comparer. Les embeddings et le LLM s'exécutent **en local**, sans clé
d'API. Une connexion Internet reste nécessaire pour scraper les pages et télécharger les modèles.

## Le workflow

```
URL
 │  1. Scraping avec Scrapy (respecte robots.txt)
 ▼
Fichier JSON dans scraped_data/
 │  2. Découpage en blocs (nœuds)
 │  3. Embedding multilingue BAAI/bge-m3
 ▼
Chroma DB dans chroma_db/
 │  4. Un routeur (llama3.2) choisit le retriever adapté à la question
 │  5. Le retriever renvoie les passages utiles
 ▼
llama3.2 (via Ollama) rédige la réponse et le projet affiche les sources
```

| Brique | Outil |
|---|---|
| Scraping | Scrapy |
| Orchestration du RAG | LlamaIndex |
| Base vectorielle | Chroma DB |
| Embedding (français, anglais et autres langues) | `BAAI/bge-m3` (HuggingFace) |
| LLM | `llama3.2` (Ollama) |
| Interfaces | Gradio ou ligne de commande |

### Embedding et LLM : deux rôles différents

- Le **modèle d'embedding** transforme les textes et la question en vecteurs pour retrouver les passages proches.
- Le **LLM** reçoit le texte de ces passages, pas les vecteurs, et rédige la réponse.

On peut donc utiliser `bge-m3` pour la recherche et `llama3.2` pour la génération. Si vous changez de modèle
d'embedding, il faut réindexer les documents.

## Les six retrievers

| Nom | Principe | Question visée |
|---|---|---|
| `vector` | Recherche par proximité de sens sur des blocs de 512 tokens | Question générale, même formulée avec d'autres mots |
| `bm25` | Recherche par mots-clés, sans embedding | Mot exact, nom propre, référence |
| `document_summary` | Choisit une page grâce à son résumé, puis renvoie tous ses blocs | Résumé ou sujet général d'une page |
| `auto_merging` | Cherche de petits blocs et les remplace par leur parent quand assez de blocs voisins sont trouvés | Explication détaillée sur plusieurs passages |
| `recursive` | Cherche de petits blocs enfants, puis suit leur référence vers le bloc parent | Fait précis avec son contexte |
| `hybrid` | Combine les scores de `vector` et `bm25`, normalisés par le meilleur score de chaque côté (poids 0,7 et 0,3) | Question mêlant sens général et termes précis |

La colonne « Question visée » décrit l'usage attendu, pas un résultat garanti.

### Ce qui est stocké dans Chroma

| Collection | Retriever | Contenu vectorisé |
|---|---|---|
| `chunks` | `vector`, `hybrid` | Blocs de 512 tokens |
| `summaries` | `document_summary` | Résumés des pages |
| `leaves` | `auto_merging` | Plus petits blocs de la hiérarchie 512, 256 et 128 tokens |
| `children` | `recursive` | Blocs enfants de 128 tokens |

`bm25` fonctionne en mémoire. Il n'utilise ni stemming ni liste de mots vides, car les deux dépendent de la langue.
Comme il compare des mots exacts, il ne fait pas le lien entre une question en français et un texte en anglais.

## Les versions de l'index

Chaque construction écrit dans son propre dossier. La version active n'est jamais modifiée.

```
chroma_db/
├── active.json     # nom de la version active et de la précédente
├── v…/             # une version : collections Chroma, blocs, résumés et manifest.json
└── v…/
```

- Le projet ne passe à la nouvelle version qu'après la réussite des six retrievers, en remplaçant `active.json` en
  une seule opération. En cas d'erreur ou de fermeture de l'application, l'ancienne version reste active.
- Les vecteurs des blocs inchangés et les résumés des pages inchangées sont réutilisés.
- Au démarrage, la version active est chargée sans vectoriser les pages et aucune construction n'est lancée. Si des
  pages ont changé depuis, l'interface le signale : cliquez sur **Re-embed**. En ligne de commande, `main.py`
  construit une version avec `--url` ou s'il n'en existe aucune.
- Les demandes reçues pendant une construction sont regroupées dans une seule construction suivante.
- Après un changement de modèle d'embedding ou de taille de blocs, l'ancienne version n'est pas chargée tant que
  vous n'avez pas cliqué sur **Re-embed**.
- Les anciennes versions et les dossiers incomplets sont supprimés automatiquement. Les fichiers créés dans
  `chroma_db/` avant ce système de versions ne sont ni réutilisés ni supprimés.

## Le routeur de requêtes

En mode `auto`, `llama3.2` lit la description de chaque retriever et choisit celui qui convient à la question
(`LLMSingleSelector` de LlamaIndex). Si sa réponse est illisible ou hors des choix proposés, le projet utilise
`vector`. Le routeur peut aussi faire un mauvais choix.

Pour modifier son comportement, changez les descriptions `ROUTES` dans `modules/router.py`. Vous pouvez également
choisir un retriever à la main.

## Structure du projet

```
├── app.py                   # Interface web Gradio
├── main.py                  # Version en ligne de commande
├── compare_retrievers.py    # Comparaison des six retrievers
├── config.py                # Modèles, tailles de blocs, paramètres de recherche, prompt
├── smoke_test.py            # Test hors ligne (modèles factices)
├── test_indexing.py         # Test hors ligne des versions et de la construction en arrière-plan
├── test_app.py              # Test hors ligne de l'interface et du scraping sans perte
├── requirements.txt         # Dépendances
└── modules/
    ├── web_spider.py        # Spider Scrapy, exécuté dans un processus séparé
    ├── data_extraction.py   # Lance le spider et enregistre le JSON
    ├── data_processing.py   # Charge les JSON et prépare Chroma
    ├── retrievers.py        # Les six retrievers, leur sauvegarde et leur chargement
    ├── versions.py          # Dossiers de versions et version active
    ├── indexer.py           # Construction en arrière-plan
    ├── router.py            # Choix du retriever
    ├── query_engine.py      # Recherche puis génération de la réponse
    └── llm_interface.py     # llama3.2 et bge-m3
```

Les fonctions nommées `create_watsonx_*` gardent leur ancien nom, mais utilisent Ollama et HuggingFace.

Le notebook `Explore Advanced Retrievers in LlamaIndex.ipynb` est un support d'apprentissage séparé. Il utilise
IBM watsonx.ai et ne fait pas partie de ce workflow.

## Installation

Prérequis :

- Python (projet utilisé avec Python 3.14)
- [Ollama](https://ollama.com), avec le modèle téléchargé :

```bash
ollama pull llama3.2
```

Depuis le dossier du projet :

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux ou macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Le premier lancement télécharge `bge-m3`, un modèle de plusieurs gigaoctets.

## Utilisation

### Interface web

```bash
python app.py
```

Ouvrez `http://127.0.0.1:7860`.

1. Dans **Process Web Page**, saisissez une URL et cliquez sur **Process Page**. La page est enregistrée, puis
   l'index est construit en arrière-plan ; le panneau de droite affiche la progression.
2. Dans **Chat**, gardez le retriever sur `auto` ou choisissez-en un.
3. Posez votre question en français ou en anglais.

Le bouton **Re-embed** lance une construction de toutes les pages enregistrées, par exemple après un échec. Le chat
reste utilisable avec l'ancienne version pendant la construction.

La réponse indique le retriever utilisé et les pages sources des passages retrouvés. Pour comparer, posez la même
question avec chacun des six retrievers.

### Ligne de commande

```bash
# Scraper une page, puis discuter
python main.py --url "https://quotes.toscrape.com/"

# Réutiliser les pages déjà scrapées avec un retriever précis
python main.py --retriever bm25
```

Tapez `exit` pour quitter.

### Comparer les retrievers

Le script de comparaison attend trois articles Wikipédia précis. Scrapez-les une fois (tapez `exit` quand
l'invite de question apparaît) :

```bash
python main.py --url "https://en.wikipedia.org/wiki/Eiffel_Tower"
python main.py --url "https://fr.wikipedia.org/wiki/Photosynth%C3%A8se"
python main.py --url "https://en.wikipedia.org/wiki/Honey_bee"
python compare_retrievers.py
```

Le script mesure la **recherche**, pas la qualité des réponses du LLM : passage attendu trouvé, part des résultats
venant de la bonne page, quantité de contexte et temps. Pour la demande de résumé, il vérifie seulement la page. Les
résultats détaillés sont enregistrés dans `comparison_results.json`. Son petit jeu de questions sert à apprendre et
ne permet pas de conclure qu'un retriever est toujours meilleur.

### Test hors ligne

```bash
python smoke_test.py
python test_indexing.py
python test_app.py
```

Il vérifie l'extraction HTML, le filtrage des adresses locales, les six retrievers et le routeur avec des modèles
factices. Il ne mesure ni la qualité des réponses ni celle de la recherche.

## Réglages utiles (`config.py`)

| Paramètre | Rôle |
|---|---|
| `LLM_MODEL_ID` | Modèle Ollama |
| `LOCAL_EMBEDDING_MODEL_ID` | Modèle d'embedding |
| `EMBEDDING_DEVICE` | `cpu` par défaut, `cuda` si la mémoire GPU le permet |
| `CHUNK_SIZES`, `CHUNK_OVERLAP` | Découpage des documents |
| `SIMILARITY_TOP_K` | Nombre de résultats pour `vector`, `bm25`, `recursive` et `hybrid` |
| `LEAF_TOP_K` | Nombre de petits blocs examinés par `auto_merging` |
| `SUMMARY_TOP_K` | Nombre de pages choisies par `document_summary` |
| `HYBRID_VECTOR_WEIGHT` | Poids de `vector` dans `hybrid` ; `bm25` reçoit le reste |
| `HYBRID_CANDIDATES` | Résultats pris de chaque côté avant la fusion |

## Limites

- **LinkedIn n'est pas pris en charge** : son `robots.txt` interdit le crawling et ses conditions d'utilisation
  interdisent le scraping.
- **Scraping simple** : une seule page par URL, sans suivre les liens ni exécuter JavaScript. Tout le texte visible
  est gardé, y compris les menus et les références, ce qui ajoute du bruit à la recherche.
- **Première construction** : le chat attend que les six retrievers soient prêts, y compris le résumé de chaque
  page. Un seul échec (par exemple Ollama arrêté) empêche l'activation de la nouvelle version.
- **Construction échouée** : son travail est perdu. La suivante ne réutilise que ce que contient la version active.
- **Espace disque** : deux versions sont conservées, plus une troisième pendant une construction.
- **Lenteur possible** : les embeddings sur CPU et les résumés de `document_summary` prennent du temps sur de longues
  pages.
- **Réponses à vérifier** : un petit modèle comme `llama3.2` peut se tromper, même avec le bon passage.
- **Usage local par un seul utilisateur** : l'application écoute sur `127.0.0.1`. Contre le *DNS rebinding*, le
  contenu venant d'une adresse non publique est rejeté, mais la requête peut déjà avoir été envoyée. Ne l'exposez pas à des utilisateurs non fiables.

Avant de scraper un site, vérifiez ses conditions d'utilisation, même si son `robots.txt` autorise le crawling.
