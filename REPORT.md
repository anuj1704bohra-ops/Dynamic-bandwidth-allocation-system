# Deep Learning-Based Dynamic Bandwidth Allocation (DBA) for Intelligent Network Resource Management

**Course:** Computer Networks (2nd Year B.Tech, Semester 3)  
**Evaluation Framework:** Multi-Seed Discrete-Slot Queue Simulation (20 Random Seeds, 4 Scenarios)  
**Artifacts Generated:** `results/benchmark_20seeds_*.csv`, `results/benchmark_summary.json`, `results.svg`

---

## Abstract

This project studies dynamic bandwidth allocation on a shared bottleneck link with four synthetic traffic classes: interactive, web, video, and bulk. It compares a NumPy multi-layer perceptron (MLP) demand predictor with last-observed and moving-average forecasts, then uses priority-weighted water-filling to allocate bandwidth. A simplified discrete-slot model tracks finite per-class buffers, backlog, overflow drops, and estimated backlog-drain delay.

Six policies are compared over 20 random seeds and four synthetic traffic scenarios. In the standard scenario, the neural policy has the lowest mean unmet demand among the implementable policies (6.62 Mbps), though its throughput is close to the simpler baselines. Under a held-out distribution shift, its throughput is lower than the last-observed and moving-average policies. These results show scenario-specific benefits and limitations; they do not establish general superiority or live-network performance.

---

## 1. Introduction

Modern computer networks frequently multiplex diverse applications onto shared bottleneck links. Traditional static resource allocation (such as equal sharing or fixed time-division multiplexing) suffers from two fundamental inefficiencies:
1. **Capacity Stranding:** Under-utilized allocations for low-demand classes cannot be borrowed by bursty classes.
2. **Buffer Bloat & Packet Drops:** High-demand bursty traffic experiences excessive queuing delay and packet loss while adjacent channel capacity remains idle.

Dynamic Bandwidth Allocation (DBA) attempts to dynamically match link capacity to time-varying demand. However, proactive allocation requires anticipating demand in advance of the scheduling slot. This project investigates whether machine learning (a feed-forward neural network) provides measurable performance gains over classical time-series baselines (persistence and moving averages) when paired with priority-aware water-filling scheduling.

---

## 2. System Architecture & Problem Formulation

Consider a shared bottleneck link of capacity $C$ Mbps operating across discrete time slots $t \in \{1, 2, \dots, T\}$ of duration $\Delta t = 50$ ms. The link is shared by $K = 4$ traffic classes:
- **Interactive ($c_0$):** Low-bandwidth, ultra-low delay tolerance (VoIP, teleconferencing; weight $w_0 = 2.2$).
- **Web Browsing ($c_1$):** Bursty, transaction-oriented HTTP traffic (weight $w_1 = 1.5$).
- **Video Streaming ($c_2$):** High average bandwidth, smooth variations (weight $w_2 = 1.2$).
- **Bulk Transfer ($c_3$):** Elastic, delay-tolerant large file downloads (weight $w_3 = 0.8$).

Given the historical demand vectors of all classes over the preceding $H = 8$ slots:
$$\mathbf{X}_t = [\mathbf{d}_{t-H}, \dots, \mathbf{d}_{t-1}] \in \mathbb{R}^{H \times K}$$

The objective is to forecast next-slot demand $\hat{\mathbf{d}}_t \in \mathbb{R}^K$ and assign bandwidth allocations $\mathbf{a}_t = [a_{0,t}, \dots, a_{K-1,t}]$ subject to:
$$\sum_{i=0}^{K-1} a_{i,t} \le C, \quad a_{i,t} \ge 0$$

---

## 3. Forecasting Models and Allocation Policies

### 3.1 Demand Predictors
1. **Last-Observed Predictor (Persistence Baseline):**
   $$\hat{\mathbf{d}}_t^{\text{last}} = \mathbf{d}_{t-1}$$
   *Assumption:* Traffic exhibits high temporal continuity; the most recent sample is the best estimate.
2. **Moving-Average Predictor ($W=8$ Baseline):**
   $$\hat{\mathbf{d}}_t^{\text{ma}} = \frac{1}{W} \sum_{k=1}^W \mathbf{d}_{t-k}$$
   *Assumption:* Traffic consists of stationary signals corrupted by zero-mean noise.
3. **Neural Demand MLP (Proposed Predictor):**
   A fully-connected feed-forward neural network with architecture $32 \rightarrow 48 \rightarrow 24 \rightarrow 4$:
   $$\mathbf{h}_1 = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{h}_2 = \max(0, \mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2), \quad \hat{\mathbf{d}} = \max(0, \mathbf{h}_2 \mathbf{W}_3 + \mathbf{b}_3)$$
   Trained from scratch using NumPy with mini-batch Adam optimization ($\alpha=0.002, \beta_1=0.9, \beta_2=0.999$, MSE loss).
4. **Oracle Reference (Ideal Upper Bound):**
   $$\hat{\mathbf{d}}_t^{\text{oracle}} = \mathbf{d}_t$$
   *Assumption:* Ideal clairvoyant scheduler that possesses exact future demand. This serves solely as a theoretical benchmark.

### 3.2 Bandwidth Allocation Policies
- **Equal Share:** Assigns $a_i = C / K$ statically.
- **Max-Min Fairness:** Classical bottleneck max-min fair allocation without priority weights, satisfying smaller demands first and dividing residual capacity equally. Uses observed demand to eliminate future information leakage.
- **Priority-Weighted Water-Filling:** Allocates capacity proportionally to priority weights $w_i$, capped at predicted demand $\hat{d}_{i,t}$. Saturated classes are capped, and remaining capacity is recursively water-filled to active unsatisfied classes. Evaluated across Last-Observed, Moving Average, Neural, and Oracle forecasts.

---

## 4. Discrete-Slot Queue Model and Metrics

The simulator uses a simplified fluid queue for each traffic class. It models traffic in Megabits per slot and uses a drop-tail buffer with a default capacity of 5 Mb per class:

1. **Buffer Admission (Megabits):** At slot duration $\Delta t$, incoming bits $A_{i,t} = d_{i,t} \times \Delta t$. For buffer capacity $B = 5.0$ Mb:
   $$\text{Room}_{i,t} = \max(0, B - Q_{i,t-1})$$
   $$\text{Dropped Bits } D_{i,t} = \max(0, A_{i,t} - \text{Room}_{i,t}), \quad A'_{i,t} = A_{i,t} - D_{i,t}$$
2. **Buffer Service:** With scheduled budget $S_{i,t} = a_{i,t} \times \Delta t$:
   $$\text{Served Bits } \Omega_{i,t} = \min(Q_{i,t-1} + A'_{i,t}, S_{i,t})$$
   $$\text{Ending Backlog } Q_{i,t} = Q_{i,t-1} + A'_{i,t} - \Omega_{i,t}$$
3. **Estimated Backlog-Drain Delay:**
   $$\text{Delay}_{i,t} = \begin{cases} 0 \text{ ms}, & Q_{i,t} \le 10^{-9} \\ \frac{Q_{i,t}}{a_{i,t}} \times 1000 \text{ ms}, & Q_{i,t} > 10^{-9}, a_{i,t} > 10^{-6} \\ \frac{Q_{i,t}}{1\text{ Mbps}} \times 1000 \text{ ms}, & Q_{i,t} > 10^{-9}, a_{i,t} \le 10^{-6} \end{cases}$$
   This is an estimate from the ending backlog and scheduled service rate. It is not a packet-level or user-measured latency distribution. When no service is scheduled for a non-empty queue, the implementation uses a nominal 1 Mbps drain rate to produce a finite estimate.
4. **Jain's Fairness Index:** Computed from the per-class served rates; values closer to 1 indicate a more equal distribution.
   $$\mathcal{J}(\mathbf{x}) = \frac{\left(\sum_{i=1}^K x_i\right)^2}{K \sum_{i=1}^K x_i^2} \quad \in [1/K, 1.0]$$

Unmet demand is requested traffic not served during the slot; it can include traffic that remains queued as well as traffic dropped at the buffer limit. Drops are reported as an equivalent traffic rate in Mbps, not as a packet count. Queue pressure remains a dimensionless unmet-demand proxy and is distinct from the delay estimate.

---

## 5. Measured Experimental Results

Each scenario uses 20 independent seeds (101–120). Each seed generates 600 slots; the first eight provide forecast history, leaving 592 evaluated slots per seed (11,840 seed-slot observations per scenario). In the policy tables, each cell shows the mean of the 20 seed-level means ± the 95% t-confidence-interval half-width ($t_{0.975,19}s/\sqrt{20}$). The CSV files also contain sample standard deviations. Delay values are backlog-drain estimates from the queue model described above.

### 5.1 Scenario 1: Standard Mixed Traffic (Link Capacity: 100 Mbps)

| Policy | Throughput (Mbps) | Unmet Demand (Mbps) | Utilization (%) | Jain Fairness | Est. Delay (ms) | Dropped Rate (Mbps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Share** | $70.16 \pm 0.89$ | $7.64 \pm 0.43$ | $70.2\% \pm 0.9\%$ | $0.847 \pm 0.005$ | $38.7 \pm 1.8$ | $5.07 \pm 0.34$ |
| **Max-Min Fairness** | $73.84 \pm 0.95$ | $7.25 \pm 0.26$ | $73.8\% \pm 0.9\%$ | $0.813 \pm 0.005$ | $125.5 \pm 12.1$ | $1.26 \pm 0.26$ |
| **Last-Observed + WF** | $73.84 \pm 0.95$ | $7.28 \pm 0.27$ | $73.8\% \pm 0.9\%$ | $0.811 \pm 0.005$ | $114.2 \pm 10.7$ | $1.30 \pm 0.26$ |
| **Moving Average + WF** | $73.82 \pm 1.07$ | $9.08 \pm 0.23$ | $73.8\% \pm 1.1\%$ | $0.825 \pm 0.005$ | $91.1 \pm 2.6$ | $1.31 \pm 0.18$ |
| **Neural Dynamic + WF** | $73.67 \pm 0.98$ | **$6.62 \pm 0.42$** | $73.7\% \pm 1.0\%$ | $0.823 \pm 0.005$ | $112.1 \pm 27.1$ | $1.47 \pm 0.38$ |
| **Oracle Reference** | $73.91 \pm 0.95$ | $1.50 \pm 0.28$ | $73.9\% \pm 0.9\%$ | $0.811 \pm 0.005$ | $118.7 \pm 15.4$ | $1.23 \pm 0.26$ |

---

### 5.2 Scenario 2: Heavy Bursts (Link Capacity: 100 Mbps)

| Policy | Throughput (Mbps) | Unmet Demand (Mbps) | Utilization (%) | Jain Fairness | Est. Delay (ms) | Dropped Rate (Mbps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Share** | $90.97 \pm 0.66$ | $40.75 \pm 1.93$ | $91.0\% \pm 0.7\%$ | $0.949 \pm 0.004$ | $100.8 \pm 2.5$ | $34.52 \pm 1.96$ |
| **Max-Min Fairness** | $92.40 \pm 0.59$ | $38.25 \pm 1.94$ | $92.4\% \pm 0.6\%$ | $0.902 \pm 0.005$ | $200.0 \pm 9.1$ | $33.02 \pm 2.08$ |
| **Last-Observed + WF** | $92.39 \pm 0.59$ | $39.10 \pm 1.97$ | $92.4\% \pm 0.6\%$ | $0.872 \pm 0.004$ | $184.3 \pm 8.5$ | $33.05 \pm 2.08$ |
| **Moving Average + WF** | **$93.42 \pm 0.57$** | $40.59 \pm 2.05$ | **$93.4\% \pm 0.6\%$** | $0.890 \pm 0.004$ | $116.6 \pm 1.6$ | $32.05 \pm 2.25$ |
| **Neural Dynamic + WF** | $91.85 \pm 0.76$ | **$38.23 \pm 1.93$** | $91.9\% \pm 0.8\%$ | $0.884 \pm 0.004$ | $177.3 \pm 8.5$ | $33.57 \pm 2.02$ |
| **Oracle Reference** | $92.44 \pm 0.59$ | $33.45 \pm 2.08$ | $92.4\% \pm 0.6\%$ | $0.872 \pm 0.004$ | $222.1 \pm 12.3$ | $32.98 \pm 2.08$ |

---

### 5.3 Scenario 3: Capacity Constrained / Congestion (Link Capacity: 75 Mbps)

| Policy | Throughput (Mbps) | Unmet Demand (Mbps) | Utilization (%) | Jain Fairness | Est. Delay (ms) | Dropped Rate (Mbps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Share** | $67.79 \pm 0.25$ | $32.21 \pm 0.61$ | $90.4\% \pm 0.3\%$ | $0.954 \pm 0.002$ | $156.6 \pm 1.0$ | $30.37 \pm 0.61$ |
| **Max-Min Fairness** | $74.72 \pm 0.07$ | $25.57 \pm 0.76$ | $99.6\% \pm 0.1\%$ | $0.921 \pm 0.003$ | $214.7 \pm 4.6$ | $23.34 \pm 0.78$ |
| **Last-Observed + WF** | $74.71 \pm 0.07$ | $25.85 \pm 0.78$ | $99.6\% \pm 0.1\%$ | $0.894 \pm 0.002$ | $159.5 \pm 3.7$ | $23.45 \pm 0.78$ |
| **Moving Average + WF** | **$74.81 \pm 0.06$** | $27.00 \pm 0.80$ | **$99.7\% \pm 0.1\%$** | $0.905 \pm 0.003$ | $155.5 \pm 2.5$ | $23.37 \pm 0.77$ |
| **Neural Dynamic + WF** | $74.36 \pm 0.15$ | **$25.22 \pm 0.78$** | $99.1\% \pm 0.2\%$ | $0.887 \pm 0.005$ | $246.0 \pm 9.7$ | $23.68 \pm 0.75$ |
| **Oracle Reference** | $74.72 \pm 0.07$ | $23.85 \pm 0.78$ | $99.6\% \pm 0.1\%$ | $0.894 \pm 0.002$ | $155.8 \pm 3.8$ | $23.43 \pm 0.78$ |

---

### 5.4 Scenario 4: Distribution Shift (Held-out Out-of-Distribution Test)

In this scenario, diurnal frequencies and class proportions were shifted (video baseline doubled to 52 Mbps, period shifted from 180 to 80 slots), while the neural model was held fixed with weights trained solely on the standard distribution.

| Policy | Throughput (Mbps) | Unmet Demand (Mbps) | Utilization (%) | Jain Fairness | Est. Delay (ms) | Dropped Rate (Mbps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal Share** | $69.60 \pm 0.74$ | $30.56 \pm 0.54$ | $69.6\% \pm 0.7\%$ | $0.861 \pm 0.004$ | $46.0 \pm 1.3$ | $28.28 \pm 0.45$ |
| **Max-Min Fairness** | $87.99 \pm 0.50$ | $15.46 \pm 0.71$ | $88.0\% \pm 0.5\%$ | $0.696 \pm 0.007$ | $218.8 \pm 13.5$ | $9.65 \pm 0.74$ |
| **Last-Observed + WF** | $87.99 \pm 0.50$ | $15.65 \pm 0.73$ | $88.0\% \pm 0.5\%$ | $0.690 \pm 0.007$ | $180.7 \pm 12.4$ | $9.71 \pm 0.73$ |
| **Moving Average + WF** | **$88.79 \pm 0.57$** | $17.16 \pm 0.81$ | **$88.8\% \pm 0.6\%$** | $0.704 \pm 0.007$ | $92.0 \pm 1.6$ | $9.02 \pm 0.79$ |
| **Neural Dynamic + WF** | $85.76 \pm 0.65$ | $16.48 \pm 0.70$ | $85.8\% \pm 0.7\%$ | $0.717 \pm 0.008$ | $128.7 \pm 21.3$ | $12.00 \pm 0.72$ |
| **Oracle Reference** | $88.05 \pm 0.50$ | $9.98 \pm 0.74$ | $88.0\% \pm 0.5\%$ | $0.690 \pm 0.007$ | $200.0 \pm 17.5$ | $9.61 \pm 0.73$ |

Forecast MAE is the mean per-slot absolute error averaged over traffic classes; here ± is the standard deviation of those per-slot errors, not a confidence interval.

| Scenario | Neural MLP (Mbps) | Last-Observed (Mbps) | Moving Average (Mbps) |
| :--- | :---: | :---: | :---: |
| Standard | $3.07 \pm 1.98$ | $3.14 \pm 2.02$ | $4.39 \pm 2.37$ |
| Heavy Bursts | $6.20 \pm 6.47$ | $6.59 \pm 5.62$ | $9.86 \pm 6.46$ |
| Capacity Constrained | $3.23 \pm 2.85$ | $2.87 \pm 2.41$ | $4.19 \pm 2.77$ |
| Distribution Shift | $4.31 \pm 3.68$ | $3.86 \pm 3.59$ | $5.74 \pm 4.13$ |

---

### 5.5 Per-Class Averages in the Standard Scenario

The following values are means across the 20 seed-level results. Per-class confidence intervals are not included in the saved benchmark files.

| Policy | Interactive (Mbps) | Web (Mbps) | Video (Mbps) | Bulk (Mbps) |
| :--- | :---: | :---: | :---: | :---: |
| Equal Share | 8.34 | 18.68 | 24.88 | 18.26 |
| Max-Min Fairness | 8.31 | 18.52 | 28.90 | 18.10 |
| Last-Observed + WF | 8.33 | 18.67 | 29.12 | 17.72 |
| Moving Average + WF | 8.28 | 18.57 | 29.23 | 17.74 |
| Neural Dynamic + WF | 8.18 | 18.57 | 29.19 | 17.73 |
| Oracle Reference | 8.34 | 18.68 | 29.14 | 17.74 |

| Policy | Interactive Delay (ms) | Web Delay (ms) | Video Delay (ms) | Bulk Delay (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Equal Share | 0.07 | 8.43 | 126.07 | 20.17 |
| Max-Min Fairness | 71.82 | 123.67 | 86.49 | 219.95 |
| Last-Observed + WF | 41.45 | 71.63 | 82.03 | 261.82 |
| Moving Average + WF | 124.99 | 81.93 | 53.02 | 104.30 |
| Neural Dynamic + WF | 210.71 | 55.90 | 59.29 | 122.45 |
| Oracle Reference | 0.00 | 52.57 | 95.41 | 326.65 |

---

## 6. Discussion and Engineering Insights

### 6.1 Where Forecasting Helps
In the standard scenario, the neural policy had the lowest mean unmet demand among the implementable policies (6.62 Mbps), compared with 7.28 Mbps for last-observed demand and 7.25 Mbps for max-min fairness. Their mean throughputs were close: 73.67 Mbps for the neural policy and about 73.84 Mbps for those two baselines. These are descriptive differences; the evaluation did not run paired significance tests.

Under heavy bursts, neural unmet demand (38.23 Mbps) was nearly the same as max-min fairness (38.25 Mbps), while moving average produced the highest throughput (93.42 Mbps). In the capacity-constrained scenario, neural unmet demand was lowest among the implementable policies (25.22 Mbps), but moving average had the highest mean throughput (74.81 Mbps). No single policy leads on every metric and scenario.

### 6.2 The Pitfall of Distribution Shift
When the traffic distribution shifted, neural mean throughput was 85.76 Mbps, below last-observed (87.99 Mbps) and moving average (88.79 Mbps). Neural forecast MAE was also higher than last-observed (4.31 versus 3.86 Mbps). This synthetic held-out scenario shows a generalization weakness for the current model. Online adaptation or a fallback policy is a possible next experiment, not something established by these results.

### 6.3 Fairness vs Quality of Service (QoS) Trade-Off
In the standard scenario, equal share had the highest Jain fairness mean (0.847) and the highest dropped-rate mean (5.07 Mbps). The neural policy’s fairness mean was 0.823. The per-class estimates do not support a claim that the selected weights guarantee low interactive delay: the neural policy’s interactive delay estimate was 210.71 ms, compared with 0.07 ms under equal share. The priority weights are configuration choices; this simulation does not enforce minimum service guarantees or latency targets.

---

## 7. Limitations and Validity Scope

1. **Synthetic Traffic Traces:** While generated using autoregressive, diurnal, and Poisson-like burst components, the traces do not capture packet-level dynamics such as TCP congestion avoidance (AIMD), slow start, or retransmission timeouts.
2. **Drop-Tail Queue Simplification:** Live networks employ Active Queue Management (AQM) such as RED (Random Early Detection) or CoDel to mitigate bufferbloat; the prototype utilizes simple drop-tail thresholds.
3. **Discrete-Slot Scheduling:** The 50 ms slot abstraction assumes instantaneous allocator convergence, neglecting control plane signaling overhead and propagation delays.

---

## 8. Conclusion

This project implemented a synthetic dynamic-bandwidth-allocation simulator and compared six policies over 20 seeds and four traffic scenarios.

Our findings conclude that:
1. Neural forecasting achieved the lowest mean unmet demand among implementable policies in the standard and capacity-constrained scenarios; under heavy bursts its result was nearly tied with max-min fairness.
2. Simple forecasting baselines remained competitive, and the neural policy had lower throughput than last-observed and moving-average policies in the distribution-shift scenario.
3. Priority-weighted water-filling affected fairness and estimated class delays, but the current settings did not guarantee low interactive delay. Queue-delay results are simplified backlog-drain estimates, and all findings are limited to synthetic traffic.

---

## References

1. **Jain, R., Chiu, D.-M., & Hawe, W. R.** (1984). *A Quantitative Measure of Fairness and Permutation for Computer Systems.* Technical Report DEC-TR-301, Digital Equipment Corporation.
2. **Bertsekas, D., & Gallager, R.** (1992). *Data Networks* (2nd ed.). Prentice-Hall. (Section 6.5: Max-Min Flow Control).
3. **Shenker, S.** (1995). Fundamental design issues for the future Internet. *IEEE Journal on Selected Areas in Communications*, 13(7), 1176–1188.
4. **Floyd, S., & Jacobson, V.** (1993). Random early detection gateways for congestion avoidance. *IEEE/ACM Transactions on Networking*, 1(4), 397–413.
5. **Kingma, D. P., & Ba, J.** (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
6. **Kramer, G., Mukherjee, B., & Pesavento, G.** (2002). IPACT: a dynamic protocol for an Ethernet PON (EPON). *IEEE Communications Magazine*, 40(2), 74–80.
