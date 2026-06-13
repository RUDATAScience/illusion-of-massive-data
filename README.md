
# Informational Health Simulator: Social Desirability Bias & The Illusion of Massive Data

This repository contains simulation code that mathematically proves how "Social Desirability Bias" (or *sontaku* / social conformity) in surveys and large-scale research structurally and non-linearly eradicates the warning signals of minorities.

In modern data science—which has become over-adapted to the "Law of Large Numbers"—this project provides computational evidence to visualize the "Informational Ill-health" lurking within the data generation process itself, advocating for a new paradigm of "Informational Health Checkups."

## 📌 Background

In the era of big data, a naive belief has become pervasive: "If the sample size is large enough, errors will disappear." However, when the gravity of "social pressure" (conformity) acts upon the respondents' decision-making processes, massive datasets no longer reflect the truth. Instead, they function as a laundering mechanism that turns a "distorted consensus" into mathematical truth.

The simulations in this project quantitatively prove the following facts:

1. **Phase Transition (The Signal Cliff)**: The moment bias exceeds a certain critical point, the "small voices" of the minority do not decrease gradually; they evaporate non-linearly.
2. **Epistemic Injustice (The Deception of Statistical Significance)**: Behind the linear improvement of the mean value, structural information distortion (KL Divergence) explodes non-linearly.
3. **Illusion of Stability (The Paradox of the Law of Large Numbers)**: Maximizing the sample size ($N=10^9$) does not guarantee reaching the truth; rather, it reinforces absolute confidence in a "false stability" contaminated by conformity.

## 🧮 Mathematical Model

An individual's final utility $U$ is defined as a linear combination of their intrinsic "true utility" and their extrinsic "social desirability utility." The choice probability is calculated via the Softmax function.

$$U_{total} = (1 - v_2)U_{true} + v_2U_{target}$$

* $v_2$: The weight of social desirability (conformity). 0 represents complete honesty, and 1 represents complete social conformity.
* $\beta$: Inverse temperature coefficient (certainty). Controls the sensitivity/sharpness of the Softmax function.

## 📊 Outputs

Executing the script creates a `comprehensive_simulation_results` directory containing the following 6 high-resolution graphs (PNGs) and their corresponding raw data (CSVs):

* **Fig 1: Survival Curve** - The steep crash (cliff) of warning signals (Rating 1) as bias increases.
* **Fig 2: Structural Alteration** - The totalitarian convergence process of distributions across 5 bias scenarios.
* **Fig 3: Mean vs KL Divergence** - The contrast between superficial mean improvements and the underlying progression of information distortion (KL distance).
* **Fig 4: Power Cliff** - Proof that statistical power for anomaly detection breaks down at the critical point.
* **Fig 5: Convergence to False Stability** - Violin plots illustrating the "false stability" brought about by expanding the sample size from $N=10^3$ to $N=10^9$.
* **Fig 6: The Violence of Absolute Numbers** - Visualizing the sheer magnitude of "SOS" signals (on a scale of 50 million people) rendered completely invisible inside a mere 5% proportion when $N=10^9$.

## 🚀 Usage

This code is optimized to run on **Google Colaboratory**.

1. Upload `informational_health_sim.py` (or the Jupyter Notebook format) to Google Colab.
2. Run all cells.
3. Upon completion, a `comprehensive_simulation_archive.zip` file containing the graphs and CSVs will automatically download via your browser.

### Note on Local Execution

When running in a local Python environment, please comment out the `from google.colab import files` line at the beginning of the script and the `files.download()` line at the end. The ZIP file will be generated in your current directory.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python informational_health_sim.py

```
