# Personal Analysis & Insights

**Author's Notes**: This document captures my understanding and exploration during the project.

---

## 🔍 Results Analysis

### Training Performance

**Final Accuracy: 78.65%**

This is reasonable for CIFAR-100:
- 100 classes with only 500 training images per class
- Small images (32×32 pixels)
- ResNet-18 is relatively shallow compared to ResNet-50

**Interesting observation**: Train accuracy hit 100% by epoch 160, but test remained at 78%. This ~21% gap shows the model is **overfitting** - memorizing training data but not generalizing perfectly.

### OOD Detection: Why Different Methods Excel on Different Datasets

**SVHN (Street Numbers):**
- Energy score wins (0.876 AUROC)
- Simple methods work well
- **My hypothesis**: SVHN is visually very different from CIFAR-100 (text vs objects), so logit-based methods easily separate them

**DTD (Textures):**
- NECO wins (0.813 AUROC)
- Feature-based methods do better
- **My hypothesis**: Textures look more "object-like" than street numbers, so we need geometric reasoning (Neural Collapse) to detect them

### The ViM Problem 🤔

**ViM performed terribly**: 0.225 AUROC on SVHN (worse than random!)

**Possible reasons I should investigate:**
1. Wrong hyperparameter α (residual scaling)
2. Maybe principal subspace calculation is wrong?
3. Need more training data to estimate null space properly?
4. Implementation bug - should double-check the paper

**Note to self**: This is actually realistic - not all methods work perfectly. Good learning experience to debug this.

---

## 🔷 Neural Collapse Deep Dive

### What I Learned About NC1

**My NC1 = 1,126,273** (very high, collapse didn't happen strongly)

At first I was worried, but after reading the papers:
- NC1 only collapses to near-zero when accuracy > 95%
- My 78.65% accuracy means classes still have variance
- This is **expected** for CIFAR-100's difficulty

### NC2 is Perfect! 🎉

**Equiangular deviation: -0.0101 (target: -0.0101)**

This is amazing - despite NC1 not collapsing, the class means form a perfect simplex structure!

**Why?**: The optimizer pushes class means apart (maximize margin) even if within-class variance isn't zero yet.

### NC3 → NC4 Connection

**NC3: Weights align with means (0.977 cosine similarity)**
**NC4: Softmax = NCC (100% agreement on training data)**

These are related: When W_c ≈ μ_c, then:
```
softmax(W^T x) ≈ softmax(μ^T x) = nearest class center
```

The math makes sense!

---

## 🧪 Experiments I Want to Try

### 1. Temperature Scaling for Energy Score
The Energy score uses:
```python
score = T * logsumexp(logits / T)
```

Default T=1. What if I try T=0.5 or T=2?

**Hypothesis**: Lower T → sharper distribution → better separation?

### 2. Fix ViM Implementation
Need to:
- Check the original ViM paper formulas
- Verify my principal subspace calculation
- Try different α values

### 3. Train Longer?
200 epochs → 78.65% accuracy
What if 400 epochs? Would NC1 collapse more?

**Trade-off**: More overfitting vs stronger Neural Collapse

---

## 📊 Failure Case Analysis

**Questions to answer:**
1. Which CIFAR-100 classes are most confused?
2. Do confused classes share visual similarity?
3. Are OOD samples from specific DTD textures harder to detect?

**Method**: Load best_model.pth, compute confusion matrix, visualize worst predictions.

---

## 💭 Connections to Theory

### Why Neural Collapse Happens

From Papyan et al. (2020), the cross-entropy loss:
```
L = -log(softmax(W_c^T h))
```

At optimum:
- Numerator W_c^T h should be large (push correct class up)
- Denominator Σ exp(W_k^T h) should be small (push wrong classes down)

This naturally leads to:
- Features h collapsing to class means (NC1)
- Means spreading apart in simplex (NC2)
- Weights aligning with means (NC3)

**Mind = blown** 🤯

### Why NECO Works for OOD

OOD samples don't fit the simplex structure:
```
In-distribution:  features ≈ one of the μ_c (close to a vertex)
Out-distribution: features in "between" vertices (violates simplex)
```

NECO measures this geometric violation!

---

## 🎯 What I Would Do Differently

1. **Start with fewer epochs** - I did 1 epoch test first (smart!), but could've tried 50 epochs before committing to 200

2. **Monitor GPU usage** - I didn't check nvidia-smi during training to see utilization

3. **Save more checkpoints** - Every 10 epochs is good, but maybe every 5 after epoch 150 to see NC emergence more granularly

4. **Implement early stopping** - Test accuracy plateaued around epoch 170, could've stopped early

---

## 📚 Key Papers I Need to Read Again

1. **Hendrycks et al. (2019)** - "Deep Anomaly Detection with Outlier Exposure"
2. **Papyan et al. (2020)** - "Prevalence of Neural Collapse during the terminal phase of deep learning training"
3. **Lee et al. (2018)** - Mahalanobis distance for OOD (to understand covariance computation)
4. **Wang et al. (2022)** - ViM paper (to fix my implementation)
5. **Sun et al. (2022)** - Energy score derivation

---

## 🤓 Personal Takeaways

**What surprised me:**
- Neural Collapse is real! Not just theory
- Simple methods (MSP, Energy) can outperform complex ones (ViM)
- Training to 100% train accuracy doesn't hurt test accuracy much

**What I still don't fully understand:**
- Why ViM failed so badly (need to debug)
- How to choose between OOD methods in practice
- The connection between NC and generalization (bonus topic)

**Skills gained:**
- Implementing research papers from scratch
- Debugging ML pipelines on GPU clusters
- Evaluating OOD detection properly (AUROC, not just accuracy)

