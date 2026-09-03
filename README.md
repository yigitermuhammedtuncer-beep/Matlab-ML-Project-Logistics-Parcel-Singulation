# Parcel Singulation on an Active-Matrix Separator (AMS) via Deep Reinforcement Learning

> **Course:** Machine Learning for Mechanical Systems (A.Y. 2025–2026)  
> **Institution:** Politecnico di Milano — MSc in Mechanical Engineering (Mechatronics & Robotics)[cite: 2]  
> **Supervisor:** Prof. Loris Roveda[cite: 2]  
> **Authors:** Yigiter Tuncer, Dorin-Mihail Dinulescu, Tomas Altermatt[cite: 2]

---

## 📌 Overview & Motivation
E-commerce parcel logistics suffers severe bottlenecks from chaotic arrival streams and variable parcel dimensions[cite: 2]. Singulation—arranging unorganized parcels into a centered, evenly spaced single file—is an essential preprocessing task prior to automated barcode scanning, weighing, and sorting[cite: 2].

Traditional cross-belt or pop-up roller sorters rely on rigid heuristics that fail under high variance or dense traffic[cite: 2]. This project implements an **Active-Matrix Separator (AMS)**: a $5 \times 5$ grid of 25 independently driven motor cells ($1.0 \, \text{m}^2$ active area)[cite: 2]. Controlling 25 continuous speeds and 25 continuous steering angles ($50\text{D}$ action space) creates a combinatorial explosion for rule-based systems[cite: 2]. We formulate the environment as a continuous **Markov Decision Process (MDP)** and train an end-to-end **Proximal Policy Optimization (PPO)** agent capable of real-time decentralized traffic regulation[cite: 2].

---

## 🎯 Functional Requirements (FRs)

| ID | Requirement | Target Metric / Evaluation Method | Outcome (Run 11) |
| :--- | :--- | :--- | :--- |
| **FR1** | **Cargo Singulation** | Cumulative accuracy (%) evaluated at line $Y = 2.0 \, \text{m}$ ($1 \, \text{m}$ downstream)[cite: 2, 3] | **100% Accuracy**[cite: 2, 3] |
| **FR2** | **Minimum Gap** | Trailing-to-leading edge distance $\ge 0.15 \, \text{m}$ (Operational: $0.89 - 1.30 \, \text{m}$)[cite: 2] | **$1.30 \, \text{m}$ avg. spacing**[cite: 2, 3] |
| **FR3** | **Collision Avoidance** | Bounding box interpenetration strictly penalized via collision counter[cite: 2] | **0 Collisions**[cite: 2] |
| **FR4** | **Centerline Evacuation** | Ejection aligned with longitudinal centerline ($X = 0.5 \, \text{m}$)[cite: 2] | **Centered single-file exit**[cite: 2] |

---

## 🧠 System Architecture & MDP Formulation

* **Physical Grid:** 25 independent motorized cells ($0.2 \times 0.2 \, \text{m}$ each) + passive outfeed conveyor ($v = 1.9 \, \text{m/s}$)[cite: 2]. Simulation step $\Delta t = 0.01 \, \text{s}$ ($100 \, \text{Hz}$)[cite: 2].
* **Observation Space ($100 \times 1$ Vector):** Bounded coordinates ($x, y$), diameters ($d$), metric longitudinal velocities ($v_y$), 10 parcel exit flags, and kinematic memory via the previous action vector ($50 \times 1$)[cite: 2].
* **Action Space ($50 \times 1$ Continuous Vector):** For each cell: steering angle $\theta \in [-\pi/4, +\pi/4] \, \text{rad}$ ($\pm 45^\circ$) and tangential drive velocity $v \in [0.5, 2.2] \, \text{m/s}$[cite: 2].
* **Policy Backbone:** PPO Actor-Critic with $[512, 256, 128]$ hidden units (He-initialized, ReLU activations)[cite: 2]. Generalized Advantage Estimation (GAE $\lambda = 0.95$, $\gamma = 0.99$, Clip factor $\epsilon = 0.2$)[cite: 2].

---

## 🛠️ Key Engineering Fixes & Reward Shaping

Reaching the gold-standard policy required eliminating subtle physics bugs and iterative reward hacking across **10 reward revisions and 15 training runs**[cite: 2, 3]:

1. **Dimensionality Bug Fix:** Raw simulation emitted forward velocity ($v_y$) in $\text{m/step}$ instead of $\text{m/s}$, causing an artificial factor of $100$ error[cite: 2, 3]. Synchronized metric units via $v_{y,\text{ms}} = v_y / \Delta t$[cite: 2, 3].
2. **"Bang-Bang" Jittering (V1–V2):** Quadratic gap penalties caused violent lateral oscillations and brake-lock[cite: 2, 3]. Replaced with linear gradients for smooth deceleration[cite: 2, 3].
3. **The "Catch-22" Paradox (V3–V4):** Agent was penalized simultaneously for slowing down and colliding with downstream parcels[cite: 2, 3]. Resolved via an adaptive `is_blocked` flag exempting obstructed parcels from minimum speed floors[cite: 2, 3].
4. **Pseudo-Collision & Wall-Hugging (V8–V10):** 1D longitudinal distance checks penalized safe parallel overtaking along the X-axis, driving the agent to wedge boxes into side walls[cite: 2, 3]. Resolved via **`diff_lane` logic**, granting velocity incentives when lateral clearance $\vert{}dx\vert{} > \text{safe}_x$[cite: 2, 3].
5. **Dynamic Centering:** Parabolic centering weight ($w_{\text{center}} = 0.3 + 1.7 \cdot \text{proximity}^2$) preserves lateral maneuvering freedom upstream while enforcing strict centering near the exit[cite: 2, 3].

---

## 📊 Experimental Results & Ablation

* **Untrained Baseline:** Random actions resulted in box jams, overlapping geometry ($\text{avg. gap} = -0.128 \, \text{m}$), and false nominal accuracy ($\sim 40\%$) driven purely by random belt drift[cite: 2, 3].
* **Final Policy (Run 11):** Achieved **100% singulation accuracy** in wave-based evaluation with $0$ collisions and an industrial buffer spacing of $1.30 \, \text{m}$[cite: 2, 3].
* **Hyperparameter Sensitivity (Premature Collapse):** Reducing PPO exploration entropy ($\text{weight} = 0.005$) and increasing update frequency ($\text{Batch} = 64$, $\text{Epoch} = 10$) caused catastrophic policy collapse[cite: 2, 3]. The agent lost lateral separation rules and converged to a severe local optimum[cite: 2, 3].
* **Physical Bottlenecks:** Stress testing with aggressive continuous inflow ($2 \text{ parcels} / 0.75 \, \text{s}$) dropped accuracy to $\sim 85.7\%$ due to spatial throughput saturation of the $1.0 \, \text{m}^2$ grid, not policy breakdown[cite: 2, 3].

---

## 📁 Repository Structure

```text
├── ML Report FINAL.pdf              # Full academic engineering report[cite: 2]
├── Project Presentation.pdf         # Summary presentation deck[cite: 3]
│
├── Final Version Complete 1.1.zip   # Gold-standard trained agent (Run 11 - Wave based)
├── Final Version Complete 1.2.zip   # Gold-standard trained agent (Run 11 - Continuous time-based)
├── All Final Codes and Professor... # Deployment scripts, environment definitions, and baselines
│
├── 1st Training Big Reward Stuck... # Iteration 1: Insufficient centering & wall-hugging failure[cite: 2, 3]
├── 5th Training Fail Data.zip       # Iteration 5: Over-penalization & metric illusion[cite: 2, 3]
├── 7th Training Centering Problem.. # Iteration 7: Incomplete lane separation[cite: 2, 3]
├── 8th Training.zip                 # Iteration 8: Intermediate reward refinement
├── 15th Last Unsuccesfull Traini... # Hyperparameter collapse stress test (low entropy)[cite: 2, 3]
│
├── All Files About Trainings.zip    # Aggregated MAT-files, training logs, and metrics
└── Videos And Log Files.zip         # 30 FPS MP4 renders comparing policies before/after training[cite: 3]

🚀 How to Run (MATLAB Reinforcement Learning Toolbox)
Extract Final Version Complete 1.1.zip (or All Final Codes...).

Open MATLAB (R2024b or later recommended with Reinforcement Learning Toolbox and Deep Learning Toolbox).

Load the pre-trained agent:

Matlab
load('trainedAgent_Run11.mat');
Run the closed-loop evaluation simulation:

Matlab
run_AMS_simulation;
(Optional) To re-render high-resolution convergence plots from logged data without re-training:

Matlab
run('Reward_Plot.m');
