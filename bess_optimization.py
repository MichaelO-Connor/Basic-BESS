import numpy as np
import pandas as pd
from pulp import *
from typing import Dict, List, Tuple, Optional
import warnings

class BESSParameters:
    """Container for BESS system parameters"""
    def __init__(self,
                 total_capacity: float,           # kWh
                 total_rated_power: float,        # kW
                 rte: float = 0.95,               # Round-trip efficiency
                 initial_soc: float = 0.5,       # Initial SoC (fraction)
                 max_soc: float = 1.0,           # Max SoC (fraction)
                 min_soc: float = 0.1,           # Min SoC (fraction)
                 daily_cycles: float = 2.0,      # Max daily cycles
                 yearly_degradation: float = 0.02,  # Annual degradation rate
                 end_of_life_soh: float = 0.8,   # End of life SoH
                 self_discharge_rate: float = 0.0001,  # Hourly self-discharge
                 replacement_cost: float = 300):  # $/kWh replacement cost

        self.total_capacity = total_capacity
        self.total_rated_power = total_rated_power
        self.rte = rte
        self.charge_efficiency = np.sqrt(rte)
        self.discharge_efficiency = np.sqrt(rte)
        self.initial_soc = initial_soc
        self.max_soc = max_soc
        self.min_soc = min_soc
        self.daily_cycles = daily_cycles
        self.yearly_degradation = yearly_degradation
        self.end_of_life_soh = end_of_life_soh
        self.self_discharge_rate = self_discharge_rate
        self.replacement_cost = replacement_cost
        self.c_rate = total_rated_power / total_capacity  # C-rate calculation

        # Degradation parameters
        self.dod_ref = 0.8  # Reference depth of discharge
        self.f_cycle = 0.0002  # Cycling degradation rate per FCE
        self.f_calendar = yearly_degradation / 8760  # Calendar aging per hour


class WholesaleOnlyArbitrage:
    """MILP model for wholesale-only battery arbitrage"""

    def __init__(self, params: BESSParameters):
        self.params = params

    def create_model(self,
                     wholesale_prices: List[float],
                     timesteps: Optional[List[int]] = None) -> LpProblem:
        """
        Create the wholesale-only arbitrage MILP model

        Args:
            wholesale_prices: List of wholesale electricity prices ($/kWh)
            timesteps: List of timestep indices (default: range(len(prices)))

        Returns:
            LpProblem: The configured MILP model
        """
        if timesteps is None:
            timesteps = list(range(len(wholesale_prices)))

        T = len(timesteps)

        # Create the model
        model = LpProblem("BESS_Wholesale_Arbitrage", LpMinimize)

        # Decision Variables
        c = {}  # Charge from grid (kWh)
        d = {}  # Discharge to grid (kWh)
        soc = {}  # State of charge (kWh)
        y_ch = {}  # Binary: charging
        y_dis = {}  # Binary: discharging

        for t in timesteps:
            c[t] = LpVariable(f"charge_{t}", lowBound=0)
            d[t] = LpVariable(f"discharge_{t}", lowBound=0)
            soc[t] = LpVariable(f"soc_{t}",
                               lowBound=self.params.min_soc * self.params.total_capacity,
                               upBound=self.params.max_soc * self.params.total_capacity)
            y_ch[t] = LpVariable(f"y_charge_{t}", cat='Binary')
            y_dis[t] = LpVariable(f"y_discharge_{t}", cat='Binary')

        # Objective Function: Minimize cost (charge cost - discharge revenue + degradation)
        degradation_cost = []
        for t in timesteps:
            # Simplified degradation cost
            throughput = (d[t] / self.params.discharge_efficiency +
                         self.params.charge_efficiency * c[t])
            deg_cost = (self.params.replacement_cost * self.params.f_cycle *
                       throughput / (2 * self.params.dod_ref))
            degradation_cost.append(deg_cost)

        model += lpSum([c[t] * wholesale_prices[t] - d[t] * wholesale_prices[t]
                       + degradation_cost[t] for t in timesteps])

        # Constraints

        # Initial SoC
        model += soc[0] == self.params.initial_soc * self.params.total_capacity

        # SoC dynamics
        for t in timesteps[1:]:
            model += (soc[t] == soc[t-1] * (1 - self.params.self_discharge_rate) +
                     self.params.charge_efficiency * c[t-1] -
                     d[t-1] / self.params.discharge_efficiency)

        # Charge/discharge rate limits
        for t in timesteps:
            model += c[t] <= self.params.total_rated_power * y_ch[t]
            model += d[t] <= self.params.total_rated_power * y_dis[t]
            model += c[t] <= self.params.c_rate * self.params.total_capacity
            model += d[t] <= self.params.c_rate * self.params.total_capacity

        # Mutual exclusivity of charge/discharge
        for t in timesteps:
            model += y_ch[t] + y_dis[t] <= 1

        # Daily throughput limits
        hours_per_day = 24
        usable_capacity = (self.params.max_soc - self.params.min_soc) * self.params.total_capacity

        for day_start in range(0, T, hours_per_day):
            day_end = min(day_start + hours_per_day, T)
            day_timesteps = timesteps[day_start:day_end]

            if day_timesteps:
                # Discharge limit
                model += lpSum([d[t] for t in day_timesteps]) <= usable_capacity * self.params.daily_cycles
                # Charge limit
                model += (lpSum([c[t] for t in day_timesteps]) <=
                         usable_capacity * self.params.daily_cycles / self.params.charge_efficiency)

        return model

    def extract_results(self, model: LpProblem, timesteps: List[int]) -> Dict:
        """Extract results from solved model"""
        results = {
            'timestep': timesteps,
            'charge': [],
            'discharge': [],
            'soc': [],
            'charge_binary': [],
            'discharge_binary': []
        }

        for t in timesteps:
            for var in model.variables():
                if var.name == f"charge_{t}":
                    results['charge'].append(var.varValue or 0)
                elif var.name == f"discharge_{t}":
                    results['discharge'].append(var.varValue or 0)
                elif var.name == f"soc_{t}":
                    results['soc'].append(var.varValue or 0)
                elif var.name == f"y_charge_{t}":
                    results['charge_binary'].append(var.varValue or 0)
                elif var.name == f"y_discharge_{t}":
                    results['discharge_binary'].append(var.varValue or 0)

        return results


class MixedArbitrage:
    """MILP model for mixed retail/wholesale arbitrage with BTM load"""

    def __init__(self, params: BESSParameters):
        self.params = params

    def create_model(self,
                     load: List[float],
                     retail_prices: List[float],
                     wholesale_prices: List[float],
                     timesteps: Optional[List[int]] = None) -> LpProblem:
        """
        Create the mixed arbitrage MILP model

        Args:
            load: List of on-site load values (kW)
            retail_prices: List of retail electricity prices ($/kWh)
            wholesale_prices: List of wholesale electricity prices ($/kWh)
            timesteps: List of timestep indices

        Returns:
            LpProblem: The configured MILP model
        """
        if timesteps is None:
            timesteps = list(range(len(load)))

        T = len(timesteps)

        # Create the model
        model = LpProblem("BESS_Mixed_Arbitrage", LpMinimize)

        # Decision Variables
        c_ret = {}  # Charge from retail (kWh)
        c_wh = {}   # Charge from wholesale (kWh)
        d_load = {} # Discharge to load (kWh)
        d_wh = {}   # Discharge to wholesale (kWh)
        soc_ret = {} # Retail-tagged SoC (kWh)
        soc_wh = {}  # Wholesale-tagged SoC (kWh)
        soc = {}     # Total SoC (kWh)

        # Binary variables
        y_ch_ret = {}  # Charging from retail
        y_ch_wh = {}   # Charging from wholesale
        y_dld = {}     # Discharging to load
        y_dwh = {}     # Discharging to wholesale

        for t in timesteps:
            # Continuous variables
            c_ret[t] = LpVariable(f"c_retail_{t}", lowBound=0)
            c_wh[t] = LpVariable(f"c_wholesale_{t}", lowBound=0)
            d_load[t] = LpVariable(f"d_load_{t}", lowBound=0)
            d_wh[t] = LpVariable(f"d_wholesale_{t}", lowBound=0)

            # SoC variables
            soc_ret[t] = LpVariable(f"soc_retail_{t}", lowBound=0)
            soc_wh[t] = LpVariable(f"soc_wholesale_{t}", lowBound=0)
            soc[t] = LpVariable(f"soc_total_{t}",
                               lowBound=self.params.min_soc * self.params.total_capacity,
                               upBound=self.params.max_soc * self.params.total_capacity)

            # Binary variables
            y_ch_ret[t] = LpVariable(f"y_ch_retail_{t}", cat='Binary')
            y_ch_wh[t] = LpVariable(f"y_ch_wholesale_{t}", cat='Binary')
            y_dld[t] = LpVariable(f"y_d_load_{t}", cat='Binary')
            y_dwh[t] = LpVariable(f"y_d_wholesale_{t}", cat='Binary')

        # Objective Function
        degradation_cost = []
        for t in timesteps:
            # Total throughput for degradation
            throughput = ((d_load[t] + d_wh[t]) / self.params.discharge_efficiency +
                         self.params.charge_efficiency * (c_ret[t] + c_wh[t]))
            deg_cost = (self.params.replacement_cost * self.params.f_cycle *
                       throughput / (2 * self.params.dod_ref))
            degradation_cost.append(deg_cost)

        # Minimize: (Load - d_load) * P_ret + c_wh * P_wh - d_wh * P_wh + degradation
        model += lpSum([
        (load[t] - d_load[t]) * retail_prices[t]         # unmet-load cost
        + c_ret[t] * retail_prices[t]                    # <<< retail charge cost
        + c_wh[t] * wholesale_prices[t]                  # wholesale charge cost
        - d_wh[t] * wholesale_prices[t]                  # wholesale revenue
        + degradation_cost[t]                            # degradation €
        for t in timesteps])

        # Constraints

        # Initial conditions
        model += soc_ret[0] == self.params.initial_soc * self.params.total_capacity * 0.5
        model += soc_wh[0] == self.params.initial_soc * self.params.total_capacity * 0.5
        model += soc[0] == soc_ret[0] + soc_wh[0]

        # SoC dynamics (split by source)
        for t in timesteps[1:]:
            # Retail SoC dynamics
            model += (soc_ret[t] == soc_ret[t-1] * (1 - self.params.self_discharge_rate) +
                     self.params.charge_efficiency * c_ret[t-1] -
                     d_load[t-1] / self.params.discharge_efficiency)

            # Wholesale SoC dynamics
            model += (soc_wh[t] == soc_wh[t-1] * (1 - self.params.self_discharge_rate) +
                     self.params.charge_efficiency * c_wh[t-1] -
                     d_wh[t-1] / self.params.discharge_efficiency)

            # Total SoC
            model += soc[t] == soc_ret[t] + soc_wh[t]

        # Energy source constraints (only retail can serve load)
        for t in timesteps[:-1]:
            model += d_load[t] <= self.params.discharge_efficiency * soc_ret[t]
            model += d_wh[t] <= self.params.discharge_efficiency * soc_wh[t]

        # Charge/discharge rate constraints (as per document)
        for t in timesteps:
            # Total charge/discharge power limits
            model += c_ret[t] + c_wh[t] <= self.params.total_rated_power
            model += d_load[t] + d_wh[t] <= self.params.total_rated_power

            # Binary activation constraints
            model += c_ret[t] <= self.params.total_rated_power * y_ch_ret[t]
            model += c_wh[t] <= self.params.total_rated_power * y_ch_wh[t]
            model += d_load[t] <= self.params.total_rated_power * y_dld[t]
            model += d_wh[t] <= self.params.total_rated_power * y_dwh[t]

        # Mutual exclusivity of charging and discharging (as per document page 6)
        for t in timesteps:
            # Can't charge from both sources at once
            model += y_ch_ret[t] + y_ch_wh[t] <= 1
            # Can't discharge to both destinations at once
            model += y_dld[t] + y_dwh[t] <= 1
            # Can't charge from retail and discharge to load
            model += y_ch_ret[t] + y_dld[t] <= 1
            # Can't charge from retail and discharge to wholesale
            model += y_ch_ret[t] + y_dwh[t] <= 1
            # Can't charge from wholesale and discharge to load
            model += y_ch_wh[t] + y_dld[t] <= 1
            # Can't charge from wholesale and discharge to wholesale
            model += y_ch_wh[t] + y_dwh[t] <= 1

        # Daily throughput limits
        hours_per_day = 24
        usable_capacity = (self.params.max_soc - self.params.min_soc) * self.params.total_capacity

        for day_start in range(0, T, hours_per_day):
            day_end = min(day_start + hours_per_day, T)
            day_timesteps = timesteps[day_start:day_end]

            if day_timesteps:
                # Total discharge limit
                model += (lpSum([d_load[t] + d_wh[t] for t in day_timesteps]) <=
                         usable_capacity * self.params.daily_cycles)
                # Total charge limit
                model += (lpSum([c_ret[t] + c_wh[t] for t in day_timesteps]) <=
                         usable_capacity * self.params.daily_cycles / self.params.charge_efficiency)

        # Load must be served constraint
        for t in timesteps:
            model += d_load[t] <= load[t]

        return model

    def extract_results(self, model: LpProblem, timesteps: List[int]) -> Dict:
        """Extract results from solved model"""
        results = {
            'timestep': timesteps,
            'charge_retail': [],
            'charge_wholesale': [],
            'discharge_load': [],
            'discharge_wholesale': [],
            'soc_retail': [],
            'soc_wholesale': [],
            'soc_total': []
        }

        for t in timesteps:
            for var in model.variables():
                if var.name == f"c_retail_{t}":
                    results['charge_retail'].append(var.varValue or 0)
                elif var.name == f"c_wholesale_{t}":
                    results['charge_wholesale'].append(var.varValue or 0)
                elif var.name == f"d_load_{t}":
                    results['discharge_load'].append(var.varValue or 0)
                elif var.name == f"d_wholesale_{t}":
                    results['discharge_wholesale'].append(var.varValue or 0)
                elif var.name == f"soc_retail_{t}":
                    results['soc_retail'].append(var.varValue or 0)
                elif var.name == f"soc_wholesale_{t}":
                    results['soc_wholesale'].append(var.varValue or 0)
                elif var.name == f"soc_total_{t}":
                    results['soc_total'].append(var.varValue or 0)

        return results


def calculate_financial_metrics(results: Dict,
                               params: BESSParameters,
                               prices_retail: Optional[List[float]] = None,
                               prices_wholesale: Optional[List[float]] = None,
                               load: Optional[List[float]] = None) -> Dict:
    """Calculate post-dispatch KPIs and financial metrics"""

    metrics = {}

    if 'discharge_load' in results:  # Mixed arbitrage
        # Aggregated energy flows
        metrics['E_load'] = sum(results['discharge_load'])
        metrics['E_wh'] = sum(results['discharge_wholesale'])
        metrics['E_chg'] = sum(results['charge_retail']) + sum(results['charge_wholesale'])

        if load:
            metrics['E_ret'] = sum([max(l - d, 0) for l, d in
                                   zip(load, results['discharge_load'])])

        # Captured price spreads
        if prices_retail:
            discharge_value = sum([d * p for d, p in
                                 zip(results['discharge_load'], prices_retail)])
            charge_cost = sum([c * p for c, p in
                             zip(results['charge_retail'], prices_retail)])
            if sum(results['discharge_load']) > 0:
                metrics['Spread_retail'] = (discharge_value - charge_cost) / sum(results['discharge_load'])

        if prices_wholesale:
            discharge_value = sum([d * p for d, p in
                                 zip(results['discharge_wholesale'], prices_wholesale)])
            charge_cost = sum([c * p for c, p in
                             zip(results['charge_wholesale'], prices_wholesale)])
            if sum(results['discharge_wholesale']) > 0:
                metrics['Spread_wholesale'] = (discharge_value - charge_cost) / sum(results['discharge_wholesale'])

        # Average daily cycles
        total_discharge = sum(results['discharge_load']) + sum(results['discharge_wholesale'])

    else:  # Wholesale only
        metrics['E_chg'] = sum(results['charge'])
        metrics['E_discharge'] = sum(results['discharge'])

        # Captured price spread
        if prices_wholesale:
            discharge_value = sum([d * p for d, p in
                                 zip(results['discharge'], prices_wholesale)])
            charge_cost = sum([c * p for c, p in
                             zip(results['charge'], prices_wholesale)])
            if sum(results['discharge']) > 0:
                metrics['Spread_wholesale'] = (discharge_value - charge_cost) / sum(results['discharge'])

        total_discharge = sum(results['discharge'])

    # Average daily cycles
    days = len(results['timestep']) / 24
    usable_capacity = (params.max_soc - params.min_soc) * params.total_capacity
    metrics['Avg_cycles'] = total_discharge / (usable_capacity * days) if days > 0 else 0

    return metrics


# Example usage function
def run_example():
    """Example of how to use the MILP models"""

    # Create BESS parameters
    params = BESSParameters(
        total_capacity=1000,      # 1 MWh
        total_rated_power=250,    # 250 kW (4-hour battery)
        rte=0.95,                  # 90% round-trip efficiency
        initial_soc=0.5,          # Start at 50% SoC
        max_soc=0.95,            # Max 95% SoC
        min_soc=0.05,            # Min 5% SoC
        daily_cycles=2.0          # Max 2 cycles per day
    )

    # Generate example price data (24 hours)
    hours = 24
    wholesale_prices = [0.03 + 0.02 * np.sin(2 * np.pi * h / 24) for h in range(hours)]
    retail_prices = [0.10 + 0.03 * np.sin(2 * np.pi * h / 24) for h in range(hours)]
    load = [100 + 50 * np.sin(2 * np.pi * (h - 6) / 24) for h in range(hours)]  # kW

    print("=== Wholesale-Only Arbitrage ===")
    # Create and solve wholesale-only model
    wholesale_model = WholesaleOnlyArbitrage(params)
    model1 = wholesale_model.create_model(wholesale_prices)
    model1.solve(PULP_CBC_CMD(msg=0))

    if model1.status == LpStatusOptimal:
        results1 = wholesale_model.extract_results(model1, list(range(hours)))
        metrics1 = calculate_financial_metrics(results1, params,
                                             prices_wholesale=wholesale_prices)
        print(f"Objective value: ${value(model1.objective):.2f}")
        print(f"Total charged: {metrics1['E_chg']:.1f} kWh")
        print(f"Total discharged: {metrics1['E_discharge']:.1f} kWh")
        print(f"Average cycles: {metrics1['Avg_cycles']:.2f}")
        print(f"Wholesale spread: ${metrics1.get('Spread_wholesale', 0):.3f}/kWh")

    print("\n=== Mixed Arbitrage (BTM) ===")
    # Create and solve mixed arbitrage model
    mixed_model = MixedArbitrage(params)
    model2 = mixed_model.create_model(load, retail_prices, wholesale_prices)
    model2.solve(PULP_CBC_CMD(msg=0))

    if model2.status == LpStatusOptimal:
        results2 = mixed_model.extract_results(model2, list(range(hours)))
        metrics2 = calculate_financial_metrics(results2, params,
                                             prices_retail=retail_prices,
                                             prices_wholesale=wholesale_prices,
                                             load=load)
        print(f"Objective value: ${value(model2.objective):.2f}")
        print(f"Energy served to load: {metrics2['E_load']:.1f} kWh")
        print(f"Energy exported: {metrics2['E_wh']:.1f} kWh")
        print(f"Total charged: {metrics2['E_chg']:.1f} kWh")
        print(f"Average cycles: {metrics2['Avg_cycles']:.2f}")
        print(f"Retail spread: ${metrics2.get('Spread_retail', 0):.3f}/kWh")
        print(f"Wholesale spread: ${metrics2.get('Spread_wholesale', 0):.3f}/kWh")


if __name__ == "__main__":
    run_example()