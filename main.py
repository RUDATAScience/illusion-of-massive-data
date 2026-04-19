import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import os
from scipy.stats import ttest_ind_from_stats, binom
from google.colab import files

# ==========================================
# [Setup] Create output directory
# ==========================================
output_dir = "comprehensive_simulation_results"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# [Definitions] Mathematical Models
# ==========================================
def u_base(i, peak):
    """Base utility function"""
    return 1.0 - 0.25 * np.abs(i - peak)

def softmax(utilities, beta):
    """Softmax function"""
    exp_u = np.exp(beta * utilities)
    return exp_u / np.sum(exp_u)

def calculate_distribution(v2, beta):
    """Calculate probability distribution based on Sontaku (v2) and Certainty (beta)"""
    options = np.array([1, 2, 3, 4, 5])
    u_true1 = u_base(options, 1) # Minority (10%) true peak
    u_true2 = u_base(options, 3) # Majority (90%) true peak
    u_sontaku = u_base(options, 4) # Sontaku target
    
    U1 = (1 - v2) * u_true1 + v2 * u_sontaku
    U2 = (1 - v2) * u_true2 + v2 * u_sontaku
    
    prob_g1 = softmax(U1, beta)
    prob_g2 = softmax(U2, beta)
    
    total_prob = 0.10 * prob_g1 + 0.90 * prob_g2
    return total_prob

def kl_divergence(p_true, p_obs):
    """Calculate KL Divergence to measure information distortion"""
    epsilon = 1e-10
    p_true = np.clip(p_true, epsilon, 1)
    p_obs = np.clip(p_obs, epsilon, 1)
    return np.sum(p_true * np.log(p_true / p_obs))

# ==========================================
# Part 1: Phase Transition (Survival Curve)
# ==========================================
print("Generating Part 1: Survival Curve...")
v2_range = np.linspace(0, 1, 100)
betas_to_test = [1.0, 3.0, 5.0, 7.0]

df_graph1 = pd.DataFrame({'v2': v2_range})
plt.figure(figsize=(10, 6))

for b in betas_to_test:
    prob_1_list = [calculate_distribution(v, b)[0] for v in v2_range]
    plt.plot(v2_range, prob_1_list, label=f'Certainty (Beta) = {b}', linewidth=2)
    df_graph1[f'Beta_{b}_Prob_Rating_1'] = prob_1_list

plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Critical Point (v2=0.5)')
plt.title("Fig 1: Survival Curve of 'Rating 1 (Warning)' with Increasing Bias (v2)", fontsize=14)
plt.xlabel('Strength of Social Desirability Bias (v2)', fontsize=12)
plt.ylabel('Observation Probability of Rating 1', fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 0.12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(os.path.join(output_dir, 'fig1_survival_curve.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
df_graph1.to_csv(os.path.join(output_dir, 'data_fig1_survival_curve.csv'), index=False)

# ==========================================
# Part 2: Structural Alteration (5 Scenarios)
# ==========================================
print("Generating Part 2: 5 Scenarios Distribution...")
fixed_beta = 5.0
scenarios = [
    (0.00, "Scenario 1: Zero Bias\n(Baseline)"),
    (0.25, "Scenario 2: Mild Bias\n(Perturbation)"),
    (0.50, "Scenario 3: Critical Point\n(Cliff Edge)"),
    (0.75, "Scenario 4: False Consensus"),
    (0.95, "Scenario 5: Totalitarian\nConvergence")
]

fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
options = [1, 2, 3, 4, 5]
df_graph2 = pd.DataFrame({'Likert_Scale': options})

for ax, (v2, title) in zip(axes, scenarios):
    dist = calculate_distribution(v2, fixed_beta)
    df_graph2[f'v2_{v2}_Prob'] = dist
    colors = ['red' if i==1 else 'orange' if i==4 else 'skyblue' for i in options]
    bars = ax.bar(options, dist, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_title(f"{title}\n(v2={v2})", fontsize=11)
    ax.set_xlabel('Likert Scale', fontsize=10)
    ax.set_xticks(options)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9)

axes[0].set_ylabel('Observation Probability', fontsize=12)
plt.suptitle(f'Fig 2: Structural Alteration of Data by Bias Level (Beta={fixed_beta})', fontsize=16, y=1.05)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'fig2_distribution_scenarios.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
df_graph2.to_csv(os.path.join(output_dir, 'data_fig2_distribution_scenarios.csv'), index=False)

# ==========================================
# Part 3: Statistical Testing & Epistemic Injustice
# ==========================================
print("Generating Part 3: Statistical Power & KL Divergence...")
N_sample = 2000
dist_base = calculate_distribution(0.0, fixed_beta)
mean_base = np.sum(options * dist_base)
std_base = np.sqrt(np.sum((options - mean_base)**2 * dist_base))

results_stat = []
v2_stat_range = np.linspace(0, 1, 101)

for v2 in v2_stat_range:
    dist = calculate_distribution(v2, fixed_beta)
    mean_val = np.sum(options * dist)
    std_val = np.sqrt(np.sum((options - mean_val)**2 * dist))
    
    _, p_value = ttest_ind_from_stats(mean_base, std_base, N_sample, mean_val, std_val, N_sample)
    prob_1 = dist[0]
    
    # Power to detect if Rating 1 exceeds 5% threshold
    threshold_count = int(N_sample * 0.05)
    detection_power = 1.0 - binom.cdf(threshold_count - 1, N_sample, prob_1)
    kl_div = kl_divergence(dist_base, dist)
    
    results_stat.append({
        "Social_Desirability_Bias_v2": np.round(v2, 2),
        "Mean_Value": mean_val,
        "t_test_p_value": p_value,
        "Prob_Rating_1": prob_1,
        "Statistical_Power": detection_power,
        "KL_Divergence": kl_div
    })

df_stat = pd.DataFrame(results_stat)
df_stat.to_csv(os.path.join(output_dir, 'data_fig3_4_statistical_tests.csv'), index=False)

# Fig 3: Mean vs KL Divergence
fig3, ax1 = plt.subplots(figsize=(8, 6))
color1 = 'tab:blue'
ax1.set_xlabel('Strength of Social Desirability Bias (v2)', fontsize=12)
ax1.set_ylabel('Observed Mean Value', color=color1, fontsize=12)
ax1.plot(df_stat["Social_Desirability_Bias_v2"], df_stat["Mean_Value"], color=color1, linewidth=3, label='Observed Mean')
ax1.tick_params(axis='y', labelcolor=color1)

ax1_2 = ax1.twinx()
color2 = 'tab:red'
ax1_2.set_ylabel('Information Distortion (KL Divergence)', color=color2, fontsize=12)
ax1_2.plot(df_stat["Social_Desirability_Bias_v2"], df_stat["KL_Divergence"], color=color2, linestyle='--', linewidth=3, label='KL Divergence')
ax1_2.tick_params(axis='y', labelcolor=color2)

ax1.set_title('Fig 3: Linear Mean Improvement vs. Non-linear Information Collapse', fontsize=14)
ax1.axvline(x=0.5, color='gray', linestyle=':', label='Critical Point (v2=0.5)')
fig3.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_mean_vs_kl.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close(fig3)

# Fig 4: Power Cliff
fig4, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(df_stat["Social_Desirability_Bias_v2"], df_stat["Statistical_Power"], color='purple', linewidth=3)
ax2.set_title(f'Fig 4: The Cliff of Signal Detection Power (N={N_sample})', fontsize=14)
ax2.set_xlabel('Strength of Social Desirability Bias (v2)', fontsize=12)
ax2.set_ylabel('Probability of Anomaly Detection (Power)', fontsize=12)
ax2.axvline(x=0.5, color='gray', linestyle=':', label='Critical Point (v2=0.5)')
ax2.fill_between(df_stat["Social_Desirability_Bias_v2"], df_stat["Statistical_Power"], alpha=0.2, color='purple')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_power_cliff.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close(fig4)

# ==========================================
# Part 4: Massive Scale Computation (10^9 Monte Carlo)
# ==========================================
print("Generating Part 4: Massive Scale Monte Carlo...")
# --- User Configurable Parameters ---
NUM_TRIALS = 1000            # Number of Monte Carlo trials
N_SCALES = [10**3, 10**5, 10**7, 10**9] # Sample Sizes
V2_TARGET = 0.50             # Sontaku at Critical Point
# ------------------------------------

prob_biased = calculate_distribution(V2_TARGET, fixed_beta)
true_mean = np.sum(options * calculate_distribution(0.0, fixed_beta))
biased_theoretical_mean = np.sum(options * prob_biased)

results_massive = []
all_means_data = {}

for N in N_SCALES:
    # O(1) calculation using Multinomial Distribution
    simulated_counts = np.random.multinomial(N, prob_biased, size=NUM_TRIALS)
    simulated_probs = simulated_counts / N
    simulated_means = np.sum(simulated_probs * options, axis=1)
    all_means_data[N] = simulated_means
    
    count_1 = simulated_counts[:, 0]
    results_massive.append({
        "Sample_Size_N": N,
        "Mean_of_Means": np.mean(simulated_means),
        "Standard_Error": np.std(simulated_means),
        "Absolute_Count_of_Rating_1": np.mean(count_1)
    })

df_massive = pd.DataFrame(results_massive)
df_massive.to_csv(os.path.join(output_dir, f'data_fig5_6_massive_scale_{NUM_TRIALS}trials.csv'), index=False)

# Fig 5: Illusion of Stability (Violin Plot)
fig5, ax5 = plt.subplots(figsize=(10, 6))
means_to_plot = [all_means_data[n] for n in N_SCALES]
parts = ax5.violinplot(means_to_plot, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor('tab:red')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

ax5.set_xticks(np.arange(1, len(N_SCALES) + 1))
ax5.set_xticklabels([f"10^{int(np.log10(n))}" for n in N_SCALES])
ax5.axhline(y=true_mean, color='blue', linestyle='--', linewidth=2, label=f'True Mean ({true_mean:.2f})')
ax5.axhline(y=biased_theoretical_mean, color='black', linestyle=':', linewidth=2, label=f'Biased Convergence ({biased_theoretical_mean:.2f})')
ax5.set_title('Fig 5: Convergence to False Stability with Massive N', fontsize=14)
ax5.set_xlabel('Sample Size N (Log Scale)', fontsize=12)
ax5.set_ylabel(f'Distribution of Observed Means ({NUM_TRIALS} Trials)', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig5_massive_scale_illusion.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close(fig5)

# Fig 6: Absolute Number of Hidden Anomalies
fig6, ax6 = plt.subplots(figsize=(10, 6))
avg_counts = [res["Absolute_Count_of_Rating_1"] for res in results_massive]
ax6.plot(np.log10(N_SCALES), avg_counts, marker='o', color='purple', linewidth=3, markersize=10)
ax6.set_yscale('log')
ax6.set_title('Fig 6: The Violence of Absolute Numbers at N=1 Billion', fontsize=14)
ax6.set_xlabel('Sample Size N (Log Scale: 10^x)', fontsize=12)
ax6.set_ylabel('Absolute Count of Rating 1 (Log Scale)', fontsize=12)
ax6.set_xticks(np.log10(N_SCALES))
ax6.set_xticklabels([f"10^{int(np.log10(n))}" for n in N_SCALES])
ax6.grid(True, which="both", ls="--", alpha=0.5)

for i, txt in enumerate(avg_counts):
    ax6.annotate(f"{int(txt):,}", (np.log10(N_SCALES)[i], avg_counts[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
                 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig6_absolute_minority_count.png'), bbox_inches='tight', dpi=300, facecolor='white')
plt.close(fig6)

# ==========================================
# Final Step: ZIP Archive and Download
# ==========================================
print("Compressing results...")
zip_filename = "comprehensive_simulation_archive"
shutil.make_archive(zip_filename, 'zip', output_dir)

print(f"✅ Calculation complete! Automatically downloading {zip_filename}.zip ...")
files.download(f"{zip_filename}.zip")
