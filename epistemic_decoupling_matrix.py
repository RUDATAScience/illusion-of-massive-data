import numpy as np
from scipy.stats import t

def calculate_epistemic_decoupling():
    """
    Computes the P-value Saturation Matrix and Epistemic Decoupling Matrix
    based on Equation (1): t = (v2 / sigma) * sqrt(N)
    """
    # Parameters
    N_list = [10**2, 10**3, 10**4, 10**5, 10**6, 10**8]
    v2_list = [0.00, 0.01, 0.05, 0.10, 0.50]
    sigma = 1.0  # Assumed standard deviation for standardized effect size

    print("=== Table 1: P-value Saturation Matrix ===")
    header = "Bias (v2) | " + " | ".join([f"N=10^{int(np.log10(n))}" for n in N_list])
    print(header)
    print("-" * len(header))
    
    for v2 in v2_list:
        p_values = []
        for N in N_list:
            if v2 == 0.0:
                p_values.append("1.000")
            else:
                d = v2 / sigma  # Cohen's d
                t_val = d * np.sqrt(N) # Test statistic
                p_val = 2 * (1 - t.cdf(t_val, df=N-1)) # Two-tailed p-value
                
                if p_val < 0.001:
                    p_values.append("0.000*")
                else:
                    p_values.append(f"{p_val:.3f}")
        
        print(f"v2={v2:<4}  | " + " | ".join([f"{p:>6}" for p in p_values]))

    print("\n" + "="*50 + "\n")

    print("=== Table 2: The Epistemic Decoupling Matrix (v2 = 0.01) ===")
    v2_target = 0.01
    d_target = v2_target / sigma
    N_list_table2 = N_list[:-1] # Exclude 10^8 for Table 2 display
    
    header2 = "Metric      | " + " | ".join([f"N=10^{int(np.log10(n))}" for n in N_list_table2])
    print(header2)
    print("-" * len(header2))
    
    p_vals_t2 = []
    d_vals_t2 = []
    
    for N in N_list_table2:
        t_val = d_target * np.sqrt(N)
        p_val = 2 * (1 - t.cdf(t_val, df=N-1))
        
        if p_val < 0.001:
            p_vals_t2.append("0.000*")
        else:
            p_vals_t2.append(f"{p_val:.3f}")
            
        d_vals_t2.append(f"{d_target:.3f}")

    print("p-value     | " + " | ".join([f"{p:>6}" for p in p_vals_t2]))
    print("Cohen's d   | " + " | ".join([f"{d:>6}" for d in d_vals_t2]))

if __name__ == "__main__":
    calculate_epistemic_decoupling()
