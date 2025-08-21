---
# 🔋 BESS Arbitrage Optimization

This repository provides an optimization framework for simulating the operation of Battery Energy Storage Systems (BESS) using Mixed-Integer Linear Programming (MILP). It supports both **wholesale-only arbitrage** and **mixed retail/wholesale arbitrage** with behind-the-meter (BTM) load.

## 🚀 Features

- MILP-based dispatch models using [PuLP](https://github.com/coin-or/pulp)
- Supports:
  - **Wholesale-Only Arbitrage**
  - **Mixed Arbitrage with BTM Load**
- Includes:
  - Battery degradation modeling (calendar + cycling)
  - Self-discharge and throughput constraints
  - Daily cycle limits
- Post-optimization KPIs:
  - Energy flows, price spreads, average cycles, cost metrics

---

## ⚙️ Installation

Install required Python packages:

```bash
pip install numpy pandas pulp
````

---

## 🧠 Model Overview

### 🔧 BESS Parameters

Configurable battery parameters include:

* `total_capacity`: Energy capacity (kWh)
* `total_rated_power`: Max charge/discharge power (kW)
* `rte`: Round-trip efficiency
* `initial_soc`: Initial state of charge (fraction)
* `max_soc` / `min_soc`: SoC bounds
* `daily_cycles`: Max daily throughput in cycles
* `yearly_degradation`: Annual calendar aging
* `replacement_cost`: Degradation cost per kWh

---

### ⚡ Arbitrage Models

#### 1. `WholesaleOnlyArbitrage`

Optimizes BESS operation using **wholesale price signals only**.

Key constraints:

* State of charge (SoC) balance
* Charge/discharge power limits
* Degradation cost from throughput
* Daily throughput and cycle constraints
* Mutual exclusivity of charge/discharge

#### 2. `MixedArbitrage`

Optimizes for both **retail and wholesale price opportunities**, with on-site (BTM) load.

Features:

* Separate charge/discharge tracking for retail vs. wholesale
* Prioritizes serving local load before export
* Manages SoC tagged by energy source
* Additional mutual exclusivity constraints

---

## 📈 Example Usage

Run the included example to simulate 24 hours:

```bash
python bess_arbitrage.py
```

This runs both arbitrage models with synthetic price/load data.

### Sample Output

```
=== Wholesale-Only Arbitrage ===
Objective value: $-X.XX
Total charged: XXX.X kWh
Total discharged: XXX.X kWh
Average cycles: X.XX
Wholesale spread: $0.XXX/kWh

=== Mixed Arbitrage (BTM) ===
Objective value: $-X.XX
Energy served to load: XXX.X kWh
Energy exported: XXX.X kWh
Total charged: XXX.X kWh
Average cycles: X.XX
Retail spread: $0.XXX/kWh
Wholesale spread: $0.XXX/kWh
```

---

## 📊 KPIs Tracked

| Metric             | Description                         |
| ------------------ | ----------------------------------- |
| `E_chg`            | Total energy charged (kWh)          |
| `E_discharge`      | Total energy discharged (kWh)       |
| `E_load`           | Energy served to on-site load (kWh) |
| `E_wh`             | Energy exported to wholesale (kWh)  |
| `Spread_retail`    | Retail arbitrage value (\$/kWh)     |
| `Spread_wholesale` | Wholesale arbitrage value (\$/kWh)  |
| `Avg_cycles`       | Average daily equivalent cycles     |

---

## 🧪 Model Components

### Class: `BESSParameters`

Container for all battery and degradation attributes used by both models.

### Class: `WholesaleOnlyArbitrage`

* Method: `create_model(prices)`
* Method: `extract_results(model, timesteps)`

### Class: `MixedArbitrage`

* Method: `create_model(load, retail_prices, wholesale_prices)`
* Method: `extract_results(model, timesteps)`

### Function: `calculate_financial_metrics(results, params, ...)`

Calculates post-dispatch KPIs and financial indicators.

---
```
