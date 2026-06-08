import matplotlib.pyplot as plt
import numpy as np

labels = ['Precision', 'Recall', 'F1']
finetune_vals = [0.7305, 0.6455, 0.6854]
baseline_vals = [0.6600, 0.6080, 0.6330]
summary_vals = [0.7284, 0.5334, 0.6158]

x = np.arange(len(labels))
width = 0.25  

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, finetune_vals, width, label='Llama 7b finetune', color='skyblue', edgecolor='black')
rects2 = ax.bar(x, baseline_vals, width, label='Llama 7b Baseline', color='salmon', edgecolor='black')
rects3 = ax.bar(x + width, summary_vals, width, label='Llama 70b', color='lightgreen', edgecolor='black')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Runs and Summary Report')
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
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

output_path = 'runs/run_comparison_graph.png'
plt.savefig(output_path)
print(f"Graph saved to {output_path}")
