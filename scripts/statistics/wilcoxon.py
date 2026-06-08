import numpy as np
import scipy.stats as stats
from pathlib import Path

path = Path("/Users/eli/research/link-kg/scripts/statistics/stat.txt")

dup_70b_short = [15.22, 13.46, 13.64, 5.88, 14.29, 6.25, 5.56]
dup_70b_long = [12.50, 18.75, 20.00, 18.75, 17.78, 14.29, 12.90, 17.39, 27.63] 

dup_full_short = [15.52, 8.89, 23.53, 16.42, 25.81, 9.68, 10.71]
dup_full_long = [14.29, 6.56, 15.09, 13.95, 5.77, 2.04, 6.06, 16.22, 2.50]  

dup_short_short = [14.55, 8.89, 8.00, 9.52, 16.67, 16.67, 23.08]
dup_short_long = [6.06, 7.69, 4.35, 6.98, 14.29, 13.33, 10.00, 17.86, 20.00]  

noise_70b_short = [6.52, 3.85, 27.27, 14.71, 19.05, 6.25, 8.33]  
noise_70b_long = [25.00, 20.00, 12.00, 6.25, 37.78, 23.81, 9.68, 13.04, 10.53]  

noise_full_short = [3.45, 4.44, 17.65, 2.99, 6.45, 3.23, 10.71]
noise_full_long = [7.14, 9.84, 9.43, 16.28, 1.92, 2.04, 3.03, 8.11, 10.00] 

noise_short_short = [5.45, 6.67, 0.00, 0.00, 10.00, 0.00, 7.69] 
noise_short_long = [12.12, 12.82, 2.17, 4.65, 0.00, 13.33, 10.00, 0.00, 25.71] 

def cohens_paired(x,y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)

def test(mname, field, g1_name, g2_name, g1_data, g2_data):
    stat, p = stats.wilcoxon(g1_data, g2_data)
    d = cohens_paired(g1_data, g2_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"[{mname} - {field}] {g1_name} vs. {g2_name} \n")
        f.write(f" p: {p:.4e} \n")
        f.write(f"d: {d:.4f} \n")
        f.write("-" *50 + "\n")

if __name__ == "__main__":
    test("short", "Node Duplication", "70B Model", "Finetuned Full", dup_70b_short, dup_full_short)
    test("short", "Node Duplication", "70B Model", "Finetuned Shortcut", dup_70b_short, dup_short_short)
    test("short", "Noisy Nodes", "70B Model", "Finetuned Full", noise_70b_short, noise_full_short)
    test("short", "Noisy Nodes", "70B Model", "Finetuned Shortcut", noise_70b_short, noise_short_short)

    test("long", "Node Duplication", "70B Model", "Finetuned Full", dup_70b_long, dup_full_long)
    test("long", "Node Duplication", "70B Model", "Finetuned Shortcut", dup_70b_long, dup_short_long)
    test("long", "Noisy Nodes", "70B Model", "Finetuned Full", noise_70b_long, noise_full_long)
    test("long", "Noisy Nodes", "70B Model", "Finetuned Shortcut", noise_70b_long, noise_short_long)