# Wood et al. (2019) AMOC 5-Box Model

A standalone Python implementation of the 5-box dynamical model of the Atlantic Meridional Overturning Circulation (AMOC) described in:

> **Wood, R. A., Rodríguez, J. M., Smith, R. S., Jackson, L. C., & Hawkins, E. (2019).**  
> *Observable, low-order dynamical controls on thresholds of the Atlantic meridional overturning circulation.*  
> **Climate Dynamics**, 53(11), 6815–6834.  
> [DOI: 10.1007/s00382-019-04956-1](https://doi.org/10.1007/s00382-019-04956-1)

---

## 1. Overview of the Model

The model partitions the global ocean into five contiguous compartments corresponding to major water masses:
- **N (North Atlantic):** Deep water formation region (NADW) and the Arctic.
- **T (Atlantic Thermocline):** Upper subtropical Atlantic.
- **S (Southern Ocean):** Fresh surface Southern Ocean and return Antarctic Intermediate Water (AAIW).
- **IP (Indo-Pacific):** Indo-Pacific thermocline (carrying warm return flow).
- **B (Bottom Water):** Southward-flowing NADW and Circumpolar Deep Water (CDW).

### Flow Formulation ($q$)
AMOC strength $q$ is governed by the North-South density gradient:
$$q = \lambda [\alpha (T_S - T_N) + \beta (S_N - S_S)]$$

Incorporating the linear North Atlantic temperature feedback $T_N = \mu q + T_0$:
$$q = \frac{\lambda}{1 + \lambda \alpha \mu} \left[\alpha(T_S - T_0) + \beta(S_N - S_S)\right]$$

### Dynamics ($dS_i / dt$)
Salt conservation determines salinity in each box via:
- Overturning advection ($q$)
- Cold Water Path ($\gamma$) vs. Warm Water Path ($1 - \gamma$)
- Subtropical gyre exchange ($K_N, K_S, K_{IP}$)
- Southern Ocean vertical mixing ($\eta$)
- Surface freshwater fluxes ($F_i$) and hosing perturbation ($H \cdot A_i$)

Both forward overturning ($q \ge 0$) and reversed circulation ($q < 0$) regimes are implemented as defined in Equations (2)–(11) of the paper.

---

## 2. Calibrations Included (Table 1)

All parameter sets from Table 1 of the paper are provided:
1. `FAMOUSA_1xCO2` (Baseline FAMOUS AOGCM pre-industrial run)
2. `FAMOUSB_1xCO2` (Updated FAMOUS pre-industrial configuration)
3. `FAMOUSB_2xCO2` (FAMOUS at $2\times\text{CO}_2$)
4. `HadGEM2AO_1xCO2` (CMIP5-generation HadGEM2-AO pre-industrial)
5. `HadGEM2AO_2xCO2` (HadGEM2-AO at $2\times\text{CO}_2$)
6. `HadGEM2AO_4xCO2` (HadGEM2-AO at $4\times\text{CO}_2$)
7. `DePreSys_1999_2008` (Observation-assimilating ocean reanalysis)

---

## 3. Directory Structure

```
Rannsoknir/WoodAMOC/
├── wood_box_model.py     # Core model classes, equations, and solver
├── simulate.py           # CLI tool to run hysteresis experiment and plot
├── plot_comparison.py    # Recreates Figure 5 multi-model comparison
├── figures/              # Output directory for generated PNG figures
└── README.md             # Model documentation
```

---

## 4. How to Run

### Run single simulation (FAMOUSA baseline):
```bash
python3 simulate.py --model FAMOUSA_1xCO2
```
This performs a quasi-equilibrium hosing experiment (ramp up $H$ to $+1.0\text{ Sv}$ at $5\times 10^{-4}\text{ Sv/yr}$, then ramp down to $-0.4\text{ Sv}$), calculates $H_{\text{crit}}$, and saves:
- `figures/amoc_hysteresis_FAMOUSA_1xCO2.png`
- `figures/salinity_evolution_FAMOUSA_1xCO2.png`

### Run all calibrations and print summary table:
```bash
python3 simulate.py --model all
```

### Reproduce Figure 5 (Threshold shifts under warming):
```bash
python3 plot_comparison.py
```
Outputs `figures/figure5_comparison.png`.

---

## 5. Typical Baseline Results (FAMOUSA)
- Initial AMOC strength: **$17.14\text{ Sv}$**
- Collapse threshold ($H_{\text{crit}}$): **$0.481\text{ Sv}$** (matches $\approx 0.48\text{ Sv}$ reported in Section 3.1)
- Recovery threshold: **$-0.147\text{ Sv}$** (recovers when $H \lesssim -0.1\text{ Sv}$)
