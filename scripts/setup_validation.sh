#!/bin/bash
# Setup script pour l'environnement de validation

echo "🔧 Setting up validation environment..."

# Créer les dossiers nécessaires
mkdir -p validation_results
mkdir -p synthetic_test_results
mkdir -p ab_test_results
mkdir -p docs

# Installer les dépendances supplémentaires
source .venv/bin/activate
uv pip install matplotlib seaborn scipy scikit-learn

# Créer un fichier de config exemple
cat > validation_config.py << EOF
# Configuration pour les tests de validation

# Épisodes de test par défaut
DEFAULT_TEST_EPISODES = [1, 5, 10, 15, 20]

# Seuils de décision
SCORE_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.7,
    'acceptable': 0.5,
    'poor': 0.4
}

# Configuration des tests synthétiques
SYNTHETIC_TEST_CONFIG = {
    'num_frames_per_test': 10,
    'image_size': (480, 640),
    'attention_sigma': 40,  # Pour les ground truth gaussiennes
}

# Configuration A/B testing
AB_TEST_CONFIG = {
    'min_episodes': 3,
    'max_episodes': 20,
    'confidence_threshold': 0.05,  # p-value pour significativité
}
EOF

echo "✅ Setup complete! Ready for validation testing."
echo ""
echo "Next steps:"
echo "1. Run synthetic tests: python scripts/run_validation.py --synthetic"
echo "2. Run A/B test: python scripts/run_validation.py --ab-test"
echo "3. View results: python scripts/view_results.py"