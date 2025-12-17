#!/bin/bash
# Verify that synced files have the correct updates on HPC

echo "=================================================="
echo "Verifying Correlation Penalty Integration"
echo "=================================================="
echo ""

# Check 1: correlation_penalty.py exists
echo "1. Checking correlation_penalty.py..."
if [ -f "correlation_penalty.py" ]; then
    echo "   ✓ correlation_penalty.py exists"
    if grep -q "compute_modulation_correlation_penalty" correlation_penalty.py; then
        echo "   ✓ compute_modulation_correlation_penalty function found"
    else
        echo "   ✗ compute_modulation_correlation_penalty function NOT found"
        exit 1
    fi
else
    echo "   ✗ correlation_penalty.py NOT found"
    exit 1
fi

# Check 2: training/losses.py has weight_correlation parameter
echo ""
echo "2. Checking training/losses.py..."
if [ -f "training/losses.py" ]; then
    echo "   ✓ training/losses.py exists"
    if grep -q "weight_correlation" training/losses.py; then
        echo "   ✓ weight_correlation parameter found"
    else
        echo "   ✗ weight_correlation parameter NOT found (OLD VERSION!)"
        echo "   >>> Need to sync training/losses.py"
        exit 1
    fi
    if grep -q "return total_loss, rms_input_loss, rms_target_loss, spectral_loss_value, mel_loss_value, corr_penalty" training/losses.py; then
        echo "   ✓ Returns 6 values (includes corr_penalty)"
    else
        echo "   ✗ Does NOT return 6 values (OLD VERSION!)"
        exit 1
    fi
else
    echo "   ✗ training/losses.py NOT found"
    exit 1
fi

# Check 3: creative_agent.py has compute_modulation_correlation
echo ""
echo "3. Checking creative_agent.py..."
if [ -f "creative_agent.py" ]; then
    echo "   ✓ creative_agent.py exists"
    if grep -q "def compute_modulation_correlation" creative_agent.py; then
        echo "   ✓ compute_modulation_correlation method found"
    else
        echo "   ✗ compute_modulation_correlation method NOT found"
        exit 1
    fi
else
    echo "   ✗ creative_agent.py NOT found"
    exit 1
fi

# Check 4: compositional_creative_agent.py updated
echo ""
echo "4. Checking compositional_creative_agent.py..."
if [ -f "compositional_creative_agent.py" ]; then
    echo "   ✓ compositional_creative_agent.py exists"
    if grep -q "from correlation_penalty import compute_modulation_correlation_penalty" compositional_creative_agent.py; then
        echo "   ✓ Imports correlation_penalty module"
    else
        echo "   ✗ Does NOT import correlation_penalty (OLD VERSION!)"
        exit 1
    fi
else
    echo "   ✗ compositional_creative_agent.py NOT found"
    exit 1
fi

# Check 5: train_simple_worker.py handles 6 return values
echo ""
echo "5. Checking train_simple_worker.py..."
if [ -f "train_simple_worker.py" ]; then
    echo "   ✓ train_simple_worker.py exists"
    if grep -q "train_loss, train_rms_input, train_rms_target, train_spectral, train_mel, train_corr_penalty, train_creative, gan_metrics = train_epoch" train_simple_worker.py; then
        echo "   ✓ Expects train_corr_penalty from train_epoch"
    else
        echo "   ✗ Does NOT expect train_corr_penalty (OLD VERSION!)"
        exit 1
    fi
    if grep -q "weight_correlation=corr_weight" train_simple_worker.py; then
        echo "   ✓ Passes weight_correlation to combined_loss"
    else
        echo "   ✗ Does NOT pass weight_correlation (OLD VERSION!)"
        exit 1
    fi
else
    echo "   ✗ train_simple_worker.py NOT found"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ ALL FILES VERIFIED!"
echo "=================================================="
echo ""
echo "Correlation penalty integration is correctly synced."
echo "Ready to start training."
echo ""
