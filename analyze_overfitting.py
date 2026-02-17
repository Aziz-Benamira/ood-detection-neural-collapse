#!/usr/bin/env python3
"""
analyze_overfitting.py

Detailed analysis of training vs test performance to detect overfitting.
Analyzes the full 200 epoch training history.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history
with open('outputs/results/training_history.json', 'r') as f:
    history = json.load(f)

train_loss = np.array(history['train_loss'])
train_acc = np.array(history['train_acc']) * 100  # Convert to percentage
test_loss = np.array(history['test_loss'])
test_acc = np.array(history['test_acc']) * 100  # Convert to percentage
epochs = np.arange(1, len(train_loss) + 1)

print("=" * 80)
print("\nOVERFITTING ANALYSIS - 200 Epochs Training")

# 1. SUMMARY STATISTICS
print("\nSUMMARY STATISTICS")

print(f"\nEpoch 1:")
print(f"  Train Loss: {train_loss[0]:.4f} | Train Acc: {train_acc[0]:.2f}%")
print(f"  Test Loss:  {test_loss[0]:.4f} | Test Acc:  {test_acc[0]:.2f}%")

print(f"\nEpoch 100:")
print(f"  Train Loss: {train_loss[99]:.4f} | Train Acc: {train_acc[99]:.2f}%")
print(f"  Test Loss:  {test_loss[99]:.4f} | Test Acc:  {test_acc[99]:.2f}%")

print(f"\nEpoch 200 (Final):")
print(f"  Train Loss: {train_loss[-1]:.4f} | Train Acc: {train_acc[-1]:.2f}%")
print(f"  Test Loss:  {test_loss[-1]:.4f} | Test Acc:  {test_acc[-1]:.2f}%")

best_epoch = np.argmax(test_acc) + 1
print(f"\nBest Test Accuracy:")
print(f"  Epoch {best_epoch}: {test_acc[best_epoch-1]:.2f}%")
print(f"  Train Acc at that epoch: {train_acc[best_epoch-1]:.2f}%")

# 2. OVERFITTING METRICS
print("\nOVERFITTING INDICATORS")

# Generalization gap
gap_epoch_1 = train_acc[0] - test_acc[0]
gap_epoch_100 = train_acc[99] - test_acc[99]
gap_epoch_200 = train_acc[-1] - test_acc[-1]
gap_best = train_acc[best_epoch-1] - test_acc[best_epoch-1]

print(f"\n1. GENERALIZATION GAP (Train Acc - Test Acc):")
print(f"   Epoch 1:   {gap_epoch_1:+6.2f}% (negative = test better, often early training)")
print(f"   Epoch 100: {gap_epoch_100:+6.2f}%")
print(f"   Epoch 200: {gap_epoch_200:+6.2f}%")
print(f"   Best Epoch ({best_epoch}): {gap_best:+6.2f}%")

if gap_epoch_200 > 20:
    print("   ⚠️  SEVERE OVERFITTING: Gap > 20%")
elif gap_epoch_200 > 10:
    print("   ⚠️  MODERATE OVERFITTING: Gap > 10%")
elif gap_epoch_200 > 5:
    print("   ✅ MILD OVERFITTING: Gap 5-10% (acceptable)")
else:
    print("   ✅ NO OVERFITTING: Gap < 5%")

# Loss divergence
loss_gap_1 = test_loss[0] - train_loss[0]
loss_gap_100 = test_loss[99] - train_loss[99]
loss_gap_200 = test_loss[-1] - train_loss[-1]

print(f"\n2. LOSS DIVERGENCE (Test Loss - Train Loss):")
print(f"   Epoch 1:   {loss_gap_1:+.4f}")
print(f"   Epoch 100: {loss_gap_100:+.4f}")
print(f"   Epoch 200: {loss_gap_200:+.4f}")

if loss_gap_200 > 0.5:
    print("   ⚠️  Test loss >> Train loss (strong overfitting)")
else:
    print("   ✅ Loss gap reasonable")

# Test accuracy plateau/decline
test_acc_peak = test_acc.max()
test_acc_final = test_acc[-1]
decline = test_acc_peak - test_acc_final

print(f"\n3. TEST ACCURACY TREND:")
print(f"   Peak:  {test_acc_peak:.2f}% (epoch {np.argmax(test_acc)+1})")
print(f"   Final: {test_acc_final:.2f}% (epoch 200)")
print(f"   Decline: {decline:.2f}%")

if decline > 2:
    print("   ⚠️  Test accuracy declining (overfitting in late epochs)")
elif decline > 0.5:
    print("   ⚡ Slight decline (minor overfitting)")
else:
    print("   ✅ No decline (stable or improving)")

# ============================================================================
# 3. PHASE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("📈 TRAINING PHASE ANALYSIS")
print("=" * 80)

# Early phase (epochs 1-50)
print("\n🌱 EARLY PHASE (Epochs 1-50):")
print(f"   Train Acc: {train_acc[0]:.2f}% → {train_acc[49]:.2f}% (+{train_acc[49]-train_acc[0]:.2f}%)")
print(f"   Test Acc:  {test_acc[0]:.2f}% → {test_acc[49]:.2f}% (+{test_acc[49]-test_acc[0]:.2f}%)")
print(f"   Gap:       {gap_epoch_1:.2f}% → {train_acc[49]-test_acc[49]:.2f}%")

# Mid phase (epochs 51-150)
print("\n📊 MID PHASE (Epochs 51-150):")
print(f"   Train Acc: {train_acc[50]:.2f}% → {train_acc[149]:.2f}% (+{train_acc[149]-train_acc[50]:.2f}%)")
print(f"   Test Acc:  {test_acc[50]:.2f}% → {test_acc[149]:.2f}% (+{test_acc[149]-test_acc[50]:.2f}%)")
print(f"   Gap:       {train_acc[50]-test_acc[50]:.2f}% → {train_acc[149]-test_acc[149]:.2f}%")

# Late phase (epochs 151-200)
print("\n🔥 LATE PHASE (Epochs 151-200):")
print(f"   Train Acc: {train_acc[150]:.2f}% → {train_acc[-1]:.2f}% (+{train_acc[-1]-train_acc[150]:.2f}%)")
print(f"   Test Acc:  {test_acc[150]:.2f}% → {test_acc[-1]:.2f}% (+{test_acc[-1]-test_acc[150]:.2f}%)")
print(f"   Gap:       {train_acc[150]-test_acc[150]:.2f}% → {gap_epoch_200:.2f}%")

# When does overfitting start?
overfitting_threshold = 10  # Gap > 10% considered overfitting
gaps = train_acc - test_acc
overfitting_start = np.where(gaps > overfitting_threshold)[0]
if len(overfitting_start) > 0:
    print(f"\n⚠️  Overfitting starts around epoch {overfitting_start[0]+1} (gap > {overfitting_threshold}%)")
else:
    print(f"\n✅ No severe overfitting detected (gap never exceeded {overfitting_threshold}%)")

# ============================================================================
# 4. DETAILED EPOCH-BY-EPOCH LOG
# ============================================================================
print("\n" + "=" * 80)
print("📝 DETAILED EPOCH-BY-EPOCH LOG (Every 10 epochs)")
print("=" * 80)
print("\nEpoch | Train Loss | Train Acc | Test Loss | Test Acc | Gap (%) | Observations")
print("-" * 90)

for i in range(0, 200, 10):
    epoch = i + 1
    gap = train_acc[i] - test_acc[i]
    
    # Observation
    obs = []
    if epoch == best_epoch:
        obs.append("🏆 BEST")
    if gap > 20:
        obs.append("⚠️ HIGH GAP")
    if epoch > 1 and test_acc[i] < test_acc[i-1]:
        obs.append("📉 Test↓")
    if train_acc[i] >= 99:
        obs.append("✨ Train~100%")
    
    obs_str = " ".join(obs) if obs else ""
    
    print(f"{epoch:5d} | {train_loss[i]:10.4f} | {train_acc[i]:8.2f}% | "
          f"{test_loss[i]:9.4f} | {test_acc[i]:7.2f}% | {gap:6.2f}% | {obs_str}")

# ============================================================================
# 5. FULL TABLE (all 200 epochs to file)
# ============================================================================
print("\n" + "=" * 80)
print("💾 SAVING FULL EPOCH TABLE")
print("=" * 80)

with open('outputs/results/overfitting_analysis.txt', 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("COMPLETE TRAINING LOG - ALL 200 EPOCHS\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"{'Epoch':>5} | {'Train Loss':>11} | {'Train Acc':>9} | "
            f"{'Test Loss':>10} | {'Test Acc':>8} | {'Gap':>7}\n")
    f.write("-" * 100 + "\n")
    
    for i in range(200):
        epoch = i + 1
        gap = train_acc[i] - test_acc[i]
        f.write(f"{epoch:5d} | {train_loss[i]:11.6f} | {train_acc[i]:8.2f}% | "
                f"{test_loss[i]:10.6f} | {test_acc[i]:7.2f}% | {gap:6.2f}%\n")

print("✅ Full table saved to: outputs/results/overfitting_analysis.txt")

# ============================================================================
# 6. IDENTIFY KEY EPOCHS
# ============================================================================
print("\n" + "=" * 80)
print("🎯 KEY EPOCHS")
print("=" * 80)

# When does train accuracy hit 90%, 95%, 99%, 100%?
milestones = [90, 95, 99, 99.9]
print("\nTrain Accuracy Milestones:")
for milestone in milestones:
    idx = np.where(train_acc >= milestone)[0]
    if len(idx) > 0:
        first_epoch = idx[0] + 1
        test_at_milestone = test_acc[idx[0]]
        print(f"  {milestone:5.1f}%: Epoch {first_epoch:3d} (Test Acc: {test_at_milestone:.2f}%)")

# When does test accuracy plateau?
# Look for when test acc changes < 0.5% over 10 epochs
print("\nTest Accuracy Plateaus:")
for i in range(10, 200, 10):
    change = abs(test_acc[i] - test_acc[i-10])
    if change < 0.5:
        print(f"  Epoch {i+1}: Test acc changed only {change:.2f}% in last 10 epochs (plateau)")
        break

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
