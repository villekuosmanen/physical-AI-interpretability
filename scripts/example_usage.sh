#!/bin/bash
# Exemples d'utilisation des outils de validation

echo "🔍 EXEMPLES D'UTILISATION - VALIDATION D'ATTENTION"
echo "================================================="
echo ""

# 1. VALIDATION RAPIDE (2 minutes)
echo "1️⃣ Validation Rapide d'un Policy:"
echo "python scripts/run_validation.py --mode quick --policy ./models/my_policy.pt"
echo ""

# 2. TESTS SYNTHÉTIQUES DÉTAILLÉS (5 minutes)
echo "2️⃣ Tests Synthétiques Complets:"
echo "python scripts/run_validation.py --mode synthetic --policy ./models/my_policy.pt --output-dir ./synthetic_results"
echo ""

# 3. A/B TEST BASELINE VS AMÉLIORATION (10-15 minutes)
echo "3️⃣ A/B Test entre Baseline et Amélioration:"
echo "python scripts/run_validation.py --mode ab-test \\"
echo "    --baseline ./models/baseline_policy.pt \\"
echo "    --improved ./models/improved_attention_policy.pt \\"
echo "    --episodes 1 5 10 15 20 \\"
echo "    --output-dir ./ab_results"
echo ""

# 4. TEST SPÉCIFIQUE AVEC CONFIG CUSTOM
echo "4️⃣ Test avec Configuration d'Attention Personnalisée:"
echo "# Créer d'abord un script Python custom:"
cat > example_custom_test.py << 'EOF'
from scripts.run_validation import load_policy, run_synthetic_tests
from src.attention_maps import ACTPolicyWithAttention

# Configuration custom pour multi-layer attention
attention_config = {
    'layers_to_capture': [-3, -2, -1],  # 3 dernières couches
    'layer_weights': [0.2, 0.3, 0.5],   # Poids croissants
    'aggregate_heads': True,             # Moyenner les têtes
}

# Charger policy avec config custom
policy = load_policy("./models/my_policy.pt", attention_config)

# Lancer tests
results = run_synthetic_tests(policy)
print(f"Score avec multi-layer: {results['overall_score']:.3f}")
EOF

echo "python example_custom_test.py"
echo ""

# 5. COMPARAISON MULTIPLE
echo "5️⃣ Comparer Plusieurs Améliorations:"
cat > compare_improvements.py << 'EOF'
from scripts.ab_test_attention import AttentionABTester

# Charger plusieurs variantes
policies = {
    'baseline': load_policy('./models/baseline.pt'),
    'multi_layer': load_policy('./models/multi_layer.pt'),
    'head_specialized': load_policy('./models/head_spec.pt'),
    'gradient_weighted': load_policy('./models/grad_weight.pt'),
}

# Comparer toutes
tester = AttentionABTester()
results = tester.compare_multiple_improvements(policies, test_episodes=[1,5,10])

# Afficher le meilleur
best_policy = max(results.items(), key=lambda x: x[1]['overall_score'])
print(f"Meilleure amélioration: {best_policy[0]} (score: {best_policy[1]['overall_score']:.3f})")
EOF

echo "python compare_improvements.py"
echo ""

# 6. VALIDATION CONTINUE (CI/CD)
echo "6️⃣ Script pour CI/CD Pipeline:"
cat > ci_validation.sh << 'EOF'
#!/bin/bash
# Script de validation pour intégration continue

POLICY_PATH=$1
THRESHOLD=0.7

echo "Running attention validation for $POLICY_PATH..."

# Lancer validation
python scripts/run_validation.py --mode quick --policy $POLICY_PATH > validation_output.txt

# Extraire le score
SCORE=$(grep "OVERALL SCORE" validation_output.txt | awk '{print $3}')

# Vérifier le seuil
if (( $(echo "$SCORE > $THRESHOLD" | bc -l) )); then
    echo "✅ Validation PASSED (score: $SCORE)"
    exit 0
else
    echo "❌ Validation FAILED (score: $SCORE < threshold: $THRESHOLD)"
    exit 1
fi
EOF

echo "chmod +x ci_validation.sh"
echo "./ci_validation.sh ./models/my_policy.pt"
echo ""

echo "================================================="
echo "💡 TIPS:"
echo "- Commencez toujours par --mode quick pour un feedback rapide"
echo "- Utilisez --mode synthetic pour comprendre les faiblesses spécifiques"
echo "- Réservez --mode ab-test pour la validation finale avant déploiement"
echo "- Les résultats sont sauvés dans --output-dir avec visualisations PNG"