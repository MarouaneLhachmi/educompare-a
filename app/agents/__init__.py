"""
Les neuf agents d'intelligence artificielle du systeme EduCompare AI.

Chaque agent est autonome, recoit en entree le resultat de l'agent (ou des
agents) qui le precede(nt) et peut etre ameliore, remplace ou reentraine
individuellement sans remettre en cause le pipeline complet — conformement a
la section 3.3.2 du rapport de conception.

    1. Extraction        — deterministe        — agent1_extraction
    2. Comprehension     — hybride (LLM)       — agent2_comprehension
    3. Decoupage         — deterministe        — agent3_decoupage
    4. Vectorisation     — deep learning       — agent4_vectorisation
    5. Recherche         — base vectorielle    — agent5_recherche
    6. Comparaison       — hybride (LLM)       — agent6_comparaison
    7. Evaluation        — machine learning    — agent7_evaluation
    8. Recommandations   — hybride (LLM)       — agent8_recommandations
    9. Rapport final     — hybride (LLM)       — agent9_rapport
"""

CATALOGUE = [
    {
        "numero": 1,
        "cle": "agent1",
        "nom": "Extraction",
        "module": "agent1_extraction",
        "nature": "Déterministe",
        "technologie": "pypdf + expressions régulières (+ OCR Tesseract si scanné)",
        "description": "Lit le document déposé et en extrait le contenu textuel exploitable.",
        "icone": "📄",
    },
    {
        "numero": 2,
        "cle": "agent2",
        "nom": "Compréhension",
        "module": "agent2_comprehension",
        "nature": "Hybride",
        "technologie": "API Gemini + repli heuristique",
        "description": "Dégage la structure pédagogique : titre, chapitres, objectifs, notions clés.",
        "icone": "🧠",
    },
    {
        "numero": 3,
        "cle": "agent3",
        "nom": "Découpage",
        "module": "agent3_decoupage",
        "nature": "Déterministe",
        "technologie": "Segmentation hiérarchique avec recouvrement",
        "description": "Découpe le cours en unités de sens cohérentes prêtes à être vectorisées.",
        "icone": "✂️",
    },
    {
        "numero": 4,
        "cle": "agent4",
        "nom": "Vectorisation",
        "module": "agent4_vectorisation",
        "nature": "Deep Learning",
        "technologie": "sentence-transformers multilingue + repli LSA (TF-IDF/SVD)",
        "description": "Transforme chaque unité de contenu en représentation vectorielle dense.",
        "icone": "🔢",
    },
    {
        "numero": 5,
        "cle": "agent5",
        "nom": "Recherche",
        "module": "agent5_recherche",
        "nature": "Déterministe",
        "technologie": "Base vectorielle FAISS (IndexFlatIP)",
        "description": "Retrouve les notions des référentiels étrangers les plus proches de chaque unité.",
        "icone": "🔎",
    },
    {
        "numero": 6,
        "cle": "agent6",
        "nom": "Comparaison",
        "module": "agent6_comparaison",
        "nature": "Hybride",
        "technologie": "Similarité cosinus + API Gemini",
        "description": "Produit la cartographie des notions communes, manquantes et excédentaires.",
        "icone": "⚖️",
    },
    {
        "numero": 7,
        "cle": "agent7",
        "nom": "Évaluation",
        "module": "agent7_evaluation",
        "nature": "Machine Learning",
        "technologie": "GradientBoostingRegressor + KMeans (scikit-learn)",
        "description": "Calcule les indicateurs de qualité pédagogique et la note globale.",
        "icone": "📊",
    },
    {
        "numero": 8,
        "cle": "agent8",
        "nom": "Recommandations",
        "module": "agent8_recommandations",
        "nature": "Hybride",
        "technologie": "Priorisation algorithmique + rédaction Gemini",
        "description": "Formule des recommandations priorisées rattachées à une notion et à un référentiel.",
        "icone": "💡",
    },
    {
        "numero": 9,
        "cle": "agent9",
        "nom": "Rapport final",
        "module": "agent9_rapport",
        "nature": "Hybride",
        "technologie": "Agrégation déterministe + synthèse Gemini",
        "description": "Agrège l'ensemble des résultats en un rapport final avec synthèse exécutive.",
        "icone": "📑",
    },
]
