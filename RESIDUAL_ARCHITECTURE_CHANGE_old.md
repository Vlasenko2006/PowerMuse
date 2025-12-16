# Residual Connection Architecture Change

## BEFORE (Buggy - with residual leakage)

### Stage 1:
```
Input:  [B, D, T]  ──┐
                     ├──> concat ──> Transformer ──> projection ──┐
Target: [B, D, T]  ──┘                                            │
                                                                  │
Input:  [B, D, T]  ──┐                                            │
                     ├──> 0.5 * (Input + Target) ─────────────────┤
Target: [B, D, T]  ──┘                                            │
                                                                  ├──> ADD ──> Output1
                                                                  │
                     Transformer output ──────────────────────────┘

Output1 = projection(transformer(concat(Input, Target))) + 0.5*(Input + Target)
```

**Result:** Output1 = 50% Input + 50% Target + small_transformation

---

### Stage 2:
```
Input:   [B, D, T]  ──┐
Output1: [B, D, T]  ──┤──> concat ──> Transformer ──> projection ──┐
Target:  [B, D, T]  ──┘                                             │
                                                                    │
Input:   [B, D, T]  ──┐                                             │
Output1: [B, D, T]  ──┤──> (Input + Output1 + Target) / 3 ─────────┤
Target:  [B, D, T]  ──┘                                             │
                                                                    ├──> ADD ──> Output2
                                                                    │
                        Transformer output ──────────────────────────┘

Output2 = projection(transformer(concat(Input, Output1, Target))) + (Input + Output1 + Target)/3
```

**Result:** Output2 = 33% Input + 33% Output1 + 33% Target + small_transformation

**Since Output1 already contains 50% Input + 50% Target:**
```
Output2 = 33% Input + 33%(0.5*Input + 0.5*Target) + 33% Target
        = 33% Input + 16.5% Input + 16.5% Target + 33% Target
        = 49.5% Input + 49.5% Target + small_transformation
```

### Problem:
**The output was almost entirely a direct mix of input and target, with minimal transformation!**

---

## AFTER (Fixed - no residual leakage)

### Stage 1:
```
Input:  [B, D, T]  ──┐
                     ├──> concat ──> Transformer ──> projection ──> Output1
Target: [B, D, T]  ──┘

Output1 = projection(transformer(concat(Input, Target)))
```

**Result:** Output1 = 100% learned_transformation(Input, Target)

---

### Stage 2:
```
Input:   [B, D, T]  ──┐
Output1: [B, D, T]  ──┤──> concat ──> Transformer ──> projection ──> Output2
Target:  [B, D, T]  ──┘

Output2 = projection(transformer(concat(Input, Output1, Target)))
```

**Result:** Output2 = 100% learned_transformation(Input, Output1, Target)

---

## Key Differences

| Aspect | BEFORE (with residuals) | AFTER (no residuals) |
|--------|------------------------|---------------------|
| **Output composition** | ~50% Input + ~50% Target + small learned | 100% learned transformation |
| **Correlation with Target** | HIGH (by design, 0.5 weight) | LOW (only if model learns it) |
| **Creativity** | LIMITED (mostly copying) | HIGH (forced to transform) |
| **Training difficulty** | EASY (residuals provide gradient flow) | HARDER (must learn from scratch) |
| **Output with zero learning** | 50% Input + 50% Target mix | Random/zero/constant |
| **Anti-correlation penalty effect** | WEAK (residuals override it) | STRONG (no escape via copying) |

---

## Why the Change?

**Problem with residuals:**
- Even with anti-correlation penalties and novelty loss, the model could output something highly correlated with the target simply because it was adding 50% target directly through residuals
- The inference showed high correlation (0.7+) with target because it was literally including 50% of target by architecture design
- Validation showed the same - the residual was ensuring output ≈ 0.5*input + 0.5*target

**Goal of the fix:**
- Force the model to genuinely **transform** the inputs rather than copying them
- Make the anti-correlation cost and novelty loss actually effective
- Achieve true creative generation rather than weighted averaging

**Trade-off:**
- Training is now harder (no residual shortcuts)
- Requires reconstruction loss > 0 to guide the model toward meaningful outputs
- But results in genuinely creative transformations when trained properly

---

## Current Training Issue (Epoch 21)

**Observation:**
```
Output : ████████████████████████████████████████████████████████████████████
  RMS Levels:  Output=0.0042 (32x quieter than target)
  Correlation: Out→Target=0.007  Out→Input=0.002
```

**Why:** With zero reconstruction loss and no residual shortcuts, the model minimizes all penalties by generating near-constant quiet audio.

**Solution:** Add small reconstruction loss (e.g., 0.01) to encourage meaningful audio generation while still allowing creative transformation.
