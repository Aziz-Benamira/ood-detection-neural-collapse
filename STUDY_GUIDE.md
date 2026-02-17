# Code Mastery Guide - Study This Before Defense

This guide helps you understand EVERY part of the codebase.
Practice explaining each section out loud.

---

## 📚 File-by-File Deep Dive

### 1. `config.py` ⚙️

**Purpose**: Centralized hyperparameters

**Key constants to memorize:**
```python
NUM_EPOCHS = 200        # Why? Standard for CIFAR-100 + Neural Collapse
BATCH_SIZE = 128        # Why? Fits in GPU memory, good for BN statistics
LEARNING_RATE = 0.1     # Why? Standard SGD init for ResNet
WEIGHT_DECAY = 5e-4     # Why? L2 regularization prevents overfitting
```

**Questions professor might ask:**
- Q: "Why learning rate 0.1?"
  A: "Standard for SGD on ImageNet/CIFAR. Higher LR (0.1) works with momentum 0.9. We use cosine annealing to decay it smoothly to near-zero."

- Q: "Why 200 epochs?"
  A: "Neural Collapse papers show it emerges in 'terminal phase' (~epoch 150-200). Need to overtrain slightly beyond convergence."

---

### 2. `src/models/resnet_cifar.py` 🏗️

**Key modification from standard ResNet:**
```python
# Standard ResNet (ImageNet):
conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)  # 224×224 → 112×112
maxpool = nn.MaxPool2d(kernel_size=3, stride=2)     # 112×112 → 56×56

# Our ResNet (CIFAR):
conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)  # 32×32 → 32×32
# NO maxpool!                                       # Keep resolution
```

**Why?** CIFAR images are tiny (32×32). Aggressive downsampling loses information.

**Architecture:**
- Layer 1: [64, 64] × 2 blocks, no stride
- Layer 2: [128, 128] × 2 blocks, stride 2 (32→16)
- Layer 3: [256, 256] × 2 blocks, stride 2 (16→8)
- Layer 4: [512, 512] × 2 blocks, stride 2 (8→4)
- AvgPool: 4×4 → 1×1 → 512-dim features

**Total parameters**: 11,220,132

**Professor's question:**
- Q: "Why not use ResNet-50?"
  A: "ResNet-18 is standard for CIFAR. ResNet-50 would overfit (more params than training samples per class: 25M vs 500)."

- Q: "Why did you modify the ResNet architecture?"
  A: "Initially tried vanilla torchvision ResNet-18 and got only 42% accuracy. Realized the 7×7 stride-2 conv + maxpool aggressively downsamples 32×32 → 8×8 before learning features. Checked the original ResNet paper (He et al., 2016, Section 4.2) - they use 3×3 conv stride-1 with no maxpool for CIFAR-10. Applied same modification, jumped to 78% accuracy. This is standard in CIFAR literature - can't use ImageNet architecture directly on tiny images."

---

### 3. `src/training/trainer.py` 🎓

**Core training loop explained:**

```python
for epoch in range(start_epoch, num_epochs):
    # 1. Training phase
    model.train()  # Enable dropout, batchnorm updates
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)  # CrossEntropyLoss
        
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
    
    # 2. Evaluation phase
    model.eval()  # Disable dropout, freeze batchnorm
    with torch.no_grad():  # No gradient computation (saves memory)
        # Test accuracy
    
    # 3. Learning rate scheduling
    scheduler.step()  # Cosine annealing: LR = 0.1 * cos(π * epoch / 200)
```

**Checkpointing strategy:**
```python
# Save every 10 epochs (for resume)
if epoch % 10 == 0:
    save_checkpoint(epoch, model, optimizer, scheduler)

# Save best model (for inference)
if test_acc > best_acc:
    save_checkpoint('best', model, ...)
```

**Professor's question:**
- Q: "Why cosine annealing?"
  A: "Smooth LR decay helps escape sharp minima → better generalization. Cosine is smooth (no sudden drops like step decay)."

---

### 4. `src/ood/scores.py` 🔍

#### **MSP (Baseline)**
```python
score_msp = max(softmax(logits))
```
**Intuition**: "How confident is the model?"
**Problem**: Overconfident on OOD samples

#### **Max Logit**
```python
score_max_logit = max(logits)  # Before softmax
```
**Intuition**: "What's the strongest raw activation?"
**Better than MSP**: Logits aren't squashed by softmax

#### **Energy**
```python
score_energy = T * log(Σ exp(logits / T))
```
**Intuition**: "Free energy" from statistical physics
**Key**: Temperature T controls smoothness
**Why it works**: OOD samples have lower energy (less concentrated)

#### **Mahalanobis**
```python
# For each class c, compute:
distance_c = (x - μ_c)^T Σ^{-1} (x - μ_c)
score = -min_c(distance_c)  # Negative of closest distance
```
**Intuition**: "How far from nearest class center?"
**Key**: Uses class covariance Σ (shared across classes)
**Why it works**: OOD features are far from all class centers

#### **ViM (Virtual Logit Matching)**
```python
# 1. Decompose feature space: x = x_principal + x_residual
x_residual = x - U U^T x  # Project away principal directions

# 2. Virtual logit:
score = α * ||x_residual|| - max(logits)
```
**Intuition**: OOD samples have large residual (don't fit principal subspace)
**Why mine failed**: Likely wrong α scaling or insufficient data

#### **NECO (Neural Collapse OOD)**
```python
# 1. Measure how well x fits simplex structure
cos_sim_best = max_c(cosine(x, μ_c))
cos_sim_second_best = second_max_c(cosine(x, μ_c))

# 2. Gap between best and second
score = cos_sim_best - cos_sim_second_best
```
**Intuition**: ID samples are close to ONE vertex; OOD samples are "in between"
**Why it works**: Leverages Neural Collapse geometry

**Professor's question:**
- Q: "Which method would you use in production?"
  A: "Energy score - simple, no extra computation (no class statistics), 0.876 AUROC on SVHN. Mahalanobis if I have memory for covariance matrix."

---

### 5. `src/neural_collapse/metrics.py` 📐

#### **NC1: Within-Class Collapse**
```python
Σ_W = within-class covariance
Σ_B = between-class covariance
NC1 = tr(Σ_W @ Σ_B^{-1}) / num_classes
```
**Goal**: NC1 → 0 (features collapse to class means)
**My result**: 1,126,273 → NOT collapsed (expected: only 78% accuracy)

#### **NC2: Simplex ETF**
```python
# Part A: Equinorm (all class means same distance from global mean)
norms = ||μ_c - μ_G|| for all c
CV = std(norms) / mean(norms)  # Coefficient of variation
```
**Goal**: CV → 0 (all norms equal)
**My result**: 0.042 ✅ Excellent!

```python
# Part B: Equiangular (all pairs of means have same angle)
cos_sim(μ_i, μ_j) = -1/(C-1) for all i ≠ j
```
**For 100 classes**: target = -1/99 = -0.0101
**My result**: -0.0101 ✅ PERFECT!

#### **NC3: Self-Duality**
```python
cos_sim(W_c, μ_c - μ_G) for all c
```
**Goal**: → 1 (weights align with class means)
**My result**: 0.977 ✅ Strong alignment

#### **NC4: NCC Simplification**
```python
# Model prediction:
y_model = argmax_c(W_c^T x)

# NCC prediction:
y_ncc = argmin_c(||x - μ_c||)  # Nearest class center

# Agreement:
accuracy = mean(y_model == y_ncc)
```
**My result**: 100% on train, 96.89% on test ✅

**Professor's question:**
- Q: "Your NC1 is high but NC2-NC4 are good. Why?"
  A: "NC2-NC4 focus on class means and weights (global structure). NC1 measures within-class variance, which stays high when accuracy isn't perfect. At 78% accuracy, samples still have spread, so NC1 doesn't collapse. But the model still pushed class means into simplex (NC2) and aligned weights (NC3)."

---

### 6. `src/utils/feature_extraction.py` 🎣

**How to extract intermediate features:**

```python
# 1. Register a hook on the layer
features = []
def hook(module, input, output):
    features.append(output)

handle = model.avgpool.register_forward_hook(hook)

# 2. Forward pass
model(images)  # Hook captures avgpool output

# 3. Clean up
handle.remove()
```

**Why hooks?** Can extract from ANY layer without modifying model forward()

**Professor's question:**
- Q: "Why extract from avgpool, not the final FC layer?"
  A: "avgpool gives 512-dim feature embedding (before classifier). These are the geometric features used in Neural Collapse analysis. FC layer outputs are logits (100-dim), already class-specific."

---

## 🎤 Defense Questions You MUST Be Ready For

### Easy Questions

**Q1: What is OOD detection?**
A: Detecting when input data comes from a different distribution than training data. Example: Model trained on animals, receives a car image.

**Q2: What is AUROC?**
A: Area Under ROC Curve. Measures separability of two distributions. 0.5 = random, 1.0 = perfect separation. It's threshold-independent (doesn't depend on choosing a cutoff).

**Q3: What is Neural Collapse?**
A: Geometric phenomenon in last layer of trained networks. Features collapse to class means (NC1), means form simplex (NC2), weights align with means (NC3), nearest-center classification works (NC4).

### Medium Questions

**Q4: Why did you choose ResNet-18?**
A: Standard for CIFAR experiments. ResNet-50 would overfit (25M params vs 50K training samples). ResNet-18 has 11M params, reasonable for this dataset.

**Q5: Explain your training schedule.**
A: 200 epochs, cosine annealing (LR: 0.1 → 0), SGD with momentum 0.9, weight decay 5e-4. Cosine decay is smooth, helps Neural Collapse emerge in terminal phase.

**Q6: Why is ViM worse than random?**
A: Likely implementation issue or hyperparameter mismatch. ViM assumes OOD has larger residual in null space. Possible causes: (1) wrong α scaling, (2) insufficient training data to estimate principal subspace, (3) CIFAR features might not have strong low-rank structure.

### Hard Questions

**Q7: Your NC1 is high (1.1M) but NC2 is perfect. Explain.**
A: NC1 measures within-class variance (feature spread), which stays high at 78% accuracy. NC2 measures between-class structure (mean geometry), which can be perfect even if classes aren't collapsed. The optimizer prioritizes separating class means (NC2) over collapsing within-class variance (NC1).

**Q8: Why does NECO work for OOD detection?**
A: NECO leverages NC2 (simplex structure). In-distribution samples have features near simplex vertices (class means). OOD samples fall "in between" vertices or outside the simplex, violating the equiangular structure. NECO measures this geometric violation via cosine similarity gaps.

**Q9: What would you do differently in production?**
A: (1) Use ensemble of OOD methods, (2) Deploy Energy score (fast, no storage), (3) Add outlier exposure training, (4) Monitor OOD scores in production to detect distribution shift, (5) Retrain periodically with new data.

**Q10: How does this relate to your Gen AI work?**
A: RAG systems need OOD detection for: (1) Detecting out-of-domain queries, (2) Flagging low-quality retrievals, (3) Identifying hallucinations. Feature-based methods (Mahalanobis, NECO) could detect when embeddings don't match training corpus distribution.

---

## 🧠 Quick Quiz Yourself

Before defense, practice answering OUT LOUD:

1. What is the input/output shape at each ResNet layer?
2. What does CrossEntropyLoss = NLLLoss(LogSoftmax()) mean mathematically?
3. Why normalize CIFAR with mean=[0.5071, 0.4867, 0.4408]?
4. What happens if you forget `model.eval()` during testing?
5. Why use `with torch.no_grad()` during inference?
6. What's the difference between `scheduler.step()` before vs after optimizer.step()?
7. How does BatchNorm behave differently in train vs eval mode?
8. What's the computational complexity of Mahalanobis (O(?) per sample)?
9. Why is cosine similarity used in NECO instead of Euclidean distance?
10. What does `register_forward_hook()` do under the hood?

---

## 💯 Confidence Checklist

Before defense, check:

- [ ] Can explain every line in main.py
- [ ] Can draw ResNet-18 architecture on whiteboard
- [ ] Can derive Energy score formula from first principles
- [ ] Can explain why NC2 uses (C-1) not C in denominator
- [ ] Can justify every hyperparameter choice
- [ ] Can explain ViM failure and propose fixes
- [ ] Can answer "what would you change?" for every component
- [ ] Can connect Neural Collapse to biological neurons (if asked)
- [ ] Can relate OOD detection to real-world AI safety
- [ ] Can explain code to a non-ML engineer

---

**Study this guide daily until defense. Practice explaining out loud!**
