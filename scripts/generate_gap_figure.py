#!/usr/bin/env python3
"""
Generate "The Gap" visualization for Paper 1
Shows: Best Single → Best Strategy → Oracle progression
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from Phase 6.1 summary
data = [
    {"Track": "HC\nNon-MTL", "Best_Single": 78.51, "Best_Strategy": 80.73, "Oracle": 92.09},
    {"Track": "HC\nMTL", "Best_Single": 71.16, "Best_Strategy": 73.16, "Oracle": 89.98},
    {"Track": "MC\nNon-MTL", "Best_Single": 69.38, "Best_Strategy": 71.89, "Oracle": 87.80},
    {"Track": "MC\nMTL", "Best_Single": 69.03, "Best_Strategy": 70.24, "Oracle": 86.94},
]

df = pd.DataFrame(data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df))
width = 0.25

bars1 = ax.bar(x - width, df['Best_Single'], width, label='Best Single Model', color='#d62728')
bars2 = ax.bar(x, df['Best_Strategy'], width, label='Best Strategy (Relation-Aware)', color='#2ca02c')
bars3 = ax.bar(x + width, df['Oracle'], width, label='Oracle Upper Bound', color='#1f77b4', alpha=0.7)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Annotations for gains
for i, row in df.iterrows():
    # Strategy gain
    gain1 = row['Best_Strategy'] - row['Best_Single']
    ax.annotate(f'+{gain1:.1f}%', 
                xy=(i - width/2, row['Best_Single'] + gain1/2),
                ha='center', fontsize=8, color='green', weight='bold')
    
    # Remaining gap
    gap = row['Oracle'] - row['Best_Strategy']
    ax.annotate(f'+{gap:.1f}%\nremaining', 
                xy=(i + width/2, row['Best_Strategy'] + gap/2),
                ha='center', fontsize=7, color='gray', style='italic')

ax.set_xlabel('Track', fontsize=12)
ax.set_ylabel('LAS (%)', fontsize=12)
ax.set_title('The Gap: Achievable Gains vs Remaining Potential', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['Track'])
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(60, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/phase6.1/the_gap_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('results/phase6.1/the_gap_visualization.pdf', bbox_inches='tight')
print("✓ Saved: results/phase6.1/the_gap_visualization.png")
print("✓ Saved: results/phase6.1/the_gap_visualization.pdf")
plt.close()

# Also create a simple table version
print("\nTHE GAP - Table Format:")
print("="*80)
for _, row in df.iterrows():
    track = row['Track'].replace('\n', ' ')
    gain = row['Best_Strategy'] - row['Best_Single']
    gap = row['Oracle'] - row['Best_Strategy']
    pct_captured = (gain / (row['Oracle'] - row['Best_Single'])) * 100
    
    print(f"{track:12} | {row['Best_Single']:5.1f}% ──+{gain:4.1f}%──> "
          f"{row['Best_Strategy']:5.1f}% ──+{gap:5.1f}%──> {row['Oracle']:5.1f}% "
          f"(captured {pct_captured:4.1f}% of gap)")
print("="*80)
