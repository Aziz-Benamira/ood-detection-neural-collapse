# TODO List & Ideas

## Things I Still Need to Do

### High Priority
- [ ] Debug ViM - why AUROC = 0.225? Check paper again
- [ ] Run experiments.py and document temperature results
- [ ] Create confusion matrix visualization
- [ ] Write proper report with LaTeX

### Medium Priority
- [ ] Try training with different weight decay values
- [ ] Test with other OOD datasets (LSUN, iNaturalist?)
- [ ] Implement NC metric tracking during training (save at each epoch)
- [ ] Add dropout to ResNet? (didn't use it, is that OK?)

### Low Priority / Ideas
- [ ] Try different ResNet variants (Wide-ResNet?)
- [ ] Implement proper cross-validation
- [ ] Add learning rate warmup
- [ ] Try mixup augmentation
- [ ] Compare with other architectures (Vision Transformer?)

## Questions for Professor

1. Is 78% accuracy good enough for CIFAR-100 or should I train longer?
2. ViM failed - is it worth fixing or just mention as limitation?
3. Should I include failed experiments in report?
4. Is using existing venv OK or should I create isolated one for reproducibility?

## Weird Observations

- NC2 is perfect but NC1 is terrible - is this expected?
- Why does NECO beat Mahalanobis on DTD but not SVHN?
- Train accuracy = 100% but test = 78% - is this overfitting or normal?

## Paper Reading Notes

- **Papyan (2020)**: Need to re-read Section 3 on NC convergence rates
- **Hendrycks (2019)**: They use "outlier exposure" - should I try this?
- **Energy paper**: Formula uses T=1, but should it be tuned?

## Bugs/Issues Fixed

- ✅ Initially used vanilla ResNet-18 from torchvision → only 42% test accuracy!
- ✅ Realized 7×7 stride-2 conv + maxpool destroys 32×32 images (checked ResNet paper Section 4.2)
- ✅ Modified to 3×3 stride-1 conv, no maxpool → jumped to 78% accuracy
- ✅ Initially forgot to normalize CIFAR images
- ✅ Model training on CPU instead of GPU (needed SLURM)
- ✅ Dataloader workers=0 caused slow loading (changed to 4/8)
- ✅ Forgot model.eval() in evaluation - batchnorm was updating!

## Time Spent

- Setup & data downloading: ~1 hour
- Implementing OOD methods: ~4 hours
- Neural Collapse metrics: ~3 hours  
- Training 200 epochs: ~20 minutes
- Debugging/experiments: ~2 hours
- **Total: ~10 hours**

## Random Notes

- L40S GPU is fast! 6s per epoch
- CIFAR-100 is harder than I thought (only 78%)
- Neural Collapse is real, not just math!
- Need to practice explaining before defense
