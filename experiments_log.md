# Failed Experiments Log

## Experiment 1: Vanilla ResNet-18 (FAILED)

**Date**: Early in project  
**Goal**: Use torchvision ResNet-18 directly for CIFAR-100

**Code**:
```python
from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = nn.Linear(512, 100)  
```

**Result**: 
- Train Acc: ~60%
- Test Acc: ~42% (terrible!)

**Problem identified**:
Looked at intermediate feature map sizes:
```
Input: 32×32×3
After conv1 (7×7, stride=2): 16×16×64  ← Lost half the pixels!
After maxpool (3×3, stride=2): 8×8×64  ← Only 8×8 remaining
After layer1: 8×8×64
After layer2: 4×4×128
After layer3: 2×2×256
After layer4: 1×1×512  ← All spatial info gone
```

The aggressive downsampling destroyed spatial information before the network 
could learn useful features.

**Conclusion**: CIFAR images (32×32) are too small for ImageNet architecture.

---

## Experiment 2: ResNet-18 with CIFAR Modifications (SUCCESS)

**Hypothesis**: Maybe the first conv + maxpool are too aggressive for 32×32?

**Research**: 
- Checked original ResNet paper (He et al., 2016)
- Section 4.2 mentions CIFAR-10 experiments
- They use 3×3 conv with stride 1 (no maxpool) for CIFAR

**Code**:
```python
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()  # Remove maxpool
model.fc = nn.Linear(512, 100)
```

**Result**:
- Train Acc: 99.98%
- Test Acc: 78.35% MUCH BETTER!

**Analysis**:
Feature map sizes now:
```
Input: 32×32×3
After conv1 (3×3, stride=1): 32×32×64  ← Preserved resolution!
After maxpool (identity): 32×32×64     ← Still 32×32
After layer1: 32×32×64
After layer2: 16×16×128  ← First downsample
After layer3: 8×8×256
After layer4: 4×4×512    ← More spatial info retained
```

**Conclusion**: Adapting architecture to input size is CRITICAL.

---

## Lesson Learned

Don't blindly use architectures designed for different input sizes.
ResNet-18 was designed for ImageNet (224×224).
For CIFAR (32×32), need to:
1. Use smaller first conv kernel (3×3 instead of 7×7)
2. Reduce stride (1 instead of 2)
3. Remove maxpool
