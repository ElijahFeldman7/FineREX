import matplotlib.pyplot as plt
import numpy as np

labels = ['Precision', 'Recall', 'F1']
finetune_rel = [0.8340, 0.7286, 0.7778] 
baseline_rel = [0.4679, 0.2723, 0.3443]
summary_rel = [0.5579, 0.2713, 0.3651]

finetune_mae = 0.1071
baseline_mae = 1.2114

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, finetune_rel, width, label='Llama 7b finetune', color='skyblue', edgecolor='black')
rects2 = ax.bar(x, baseline_rel, width, label='Llama 7b Baseline', color='salmon', edgecolor='black')
rects3 = ax.bar(x + width, summary_rel, width, label='Llama 70b', color='lightgreen', edgecolor='black')

ax.set_ylabel('Scores')
ax.set_title('Relationship Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.savefig('runs/relationship_comparison_graph.png')
print("Graph saved to runs/relationship_comparison_graph.png")

# Also create a small MAE comparison
fig2, ax2 = plt.subplots(figsize=(6, 5))
mae_labels = ['Finetune', 'Baseline']
mae_vals = [finetune_mae, baseline_mae]
ax2.bar(mae_labels, mae_vals, color=['skyblue', 'salmon'], edgecolor='black', width=0.5)
ax2.set_ylabel('MAE (Lower is Better)')
ax2.set_title('Relationship Score MAE Comparison')
for i, v in enumerate(mae_vals):
    ax2.text(i, v + 0.05, f'{v:.4f}', ha='center', fontweight='bold')
ax2.set_ylim(0, max(mae_vals) * 1.2)
fig2.tight_layout()
plt.savefig('runs/relationship_mae_comparison.png')
print("Graph saved to runs/relationship_mae_comparison.png")
