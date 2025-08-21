
from bess_model_core import (
    BESSParameters,
    WholesaleOnlyArbitrage,
    MixedArbitrage,
    calculate_financial_metrics
)
import numpy as np
from pulp import LpStatusOptimal, PULP_CBC_CMD, value

def run_example():
    """Example of how to use the MILP models"""

    # Create BESS parameters
    params = BESSParameters(
        total_capacity=1000,      # 1 MWh
        total_rated_power=250,    # 250 kW (4-hour battery)
        rte=0.95,                 # 90% round-trip efficiency
        initial_soc=0.5,          # Start at 50% SoC
        max_soc=0.95,             # Max 95% SoC
        min_soc=0.05,             # Min 5% SoC
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
