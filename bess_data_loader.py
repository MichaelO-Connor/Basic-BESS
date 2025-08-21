import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import os
import pybamm

# Import the BESS optimization classes
from bess_optimization import BESSParameters, WholesaleOnlyArbitrage, calculate_financial_metrics

def simulate_battery_profile(battery_capacity_kwh, battery_power_kw, hours):
    """
    Run a basic PyBaMM simulation to generate max charge/discharge and efficiency profiles.
    Currently uses PyBaMM's default lithium-ion model for a fixed horizon.
    """
    model = pybamm.lithium_ion.SPM()
    parameter_values = model.default_parameter_values
    
    # Convert kWh to Ah: Ah = (kWh * 1000) / voltage_nominal
    nominal_voltage = 3.7
    capacity_ah = (battery_capacity_kwh * 1000) / nominal_voltage
    parameter_values["Nominal cell capacity [A.h]"] = capacity_ah

    sim = pybamm.Simulation(model, parameter_values=parameter_values)
    sim.solve([0, hours])

    soc = sim.solution["State of Charge"].data

    # Calculate daily cycles
    delta_soc = abs(np.diff(soc))
    throughput_kwh = np.sum(delta_soc) * battery_capacity_kwh
    daily_cycles = throughput_kwh / battery_capacity_kwh

    max_soc = float(np.max(soc))
    min_soc = float(np.min(soc))

    # Placeholder power and efficiency
    max_charge = [battery_power_kw] * hours
    max_discharge = [battery_power_kw] * hours
    eff_chg = [0.95] * hours
    eff_dis = [0.95] * hours

    return max_charge, max_discharge, eff_chg, eff_dis, max_soc, min_soc, daily_cycles

class BESSDataLoader:
    """Load and prepare data for BESS optimization"""
    
    def __init__(self, prices_csv_path: str, load_profile_csv_path: str = None):
        """
        Initialize data loader
        
        Args:
            prices_csv_path: Path to wholesale prices CSV file
            load_profile_csv_path: Path to industrial load profile CSV (optional)
        """
        self.prices_csv_path = prices_csv_path
        self.load_profile_csv_path = load_profile_csv_path
        self.prices_df = None
        self.load_profile_df = None
        
    def load_wholesale_prices(self, country: str = 'Germany', 
                            start_date: str = None, 
                            end_date: str = None) -> pd.DataFrame:
        """
        Load wholesale electricity prices for a specific country
        
        Args:
            country: Country name (e.g., 'Germany', 'France', etc.)
            start_date: Start date in format 'YYYY-MM-DD' (optional)
            end_date: End date in format 'YYYY-MM-DD' (optional)
            
        Returns:
            DataFrame with datetime index and price columns
        """
        print(f"Loading wholesale prices from {self.prices_csv_path}...")
        
        # Read CSV file
        df = pd.read_csv(self.prices_csv_path)
        
        # Filter for specific country
        df_country = df[df['Country'] == country].copy()
        
        if df_country.empty:
            raise ValueError(f"No data found for country: {country}")
        
        # Convert datetime strings to datetime objects
        df_country['Datetime_UTC'] = pd.to_datetime(df_country['Datetime (UTC)'], 
                                                    format='%d/%m/%Y %H:%M')
        
        # Set datetime as index
        df_country.set_index('Datetime_UTC', inplace=True)
        
        # Filter by date range if specified
        if start_date:
            df_country = df_country[df_country.index >= pd.to_datetime(start_date)]
        if end_date:
            df_country = df_country[df_country.index <= pd.to_datetime(end_date)]
        
        # Sort by datetime
        df_country.sort_index(inplace=True)
        
        # Convert EUR/MWh to $/kWh (assuming 1 EUR = 1.1 USD for example)
        # You can adjust the conversion rate as needed
        eur_to_usd = 1.1
        df_country['Price ($/kWh)'] = df_country['Price (EUR/MWhe)'] * eur_to_usd / 1000
        
        self.prices_df = df_country
        
        print(f"Loaded {len(df_country)} price records for {country}")
        print(f"Date range: {df_country.index.min()} to {df_country.index.max()}")
        print(f"Price range: ${df_country['Price ($/kWh)'].min():.4f} to ${df_country['Price ($/kWh)'].max():.4f} per kWh")
        
        return df_country
    
    def load_industrial_load_profile(self, peak_power_kw: float = 1000) -> pd.DataFrame:
        """
        Load industrial load profile data
        
        Args:
            peak_power_kw: Peak power capacity in kW
            
        Returns:
            DataFrame with load values in kW
        """
        if not self.load_profile_csv_path:
            print("No load profile path specified")
            return None
            
        print(f"Loading industrial load profile from {self.load_profile_csv_path}...")
        
        # Read CSV file
        df = pd.read_csv(self.load_profile_csv_path)
        
        # Assuming the column is named '% of Peak Power'
        if '% of Peak Power' not in df.columns:
            raise ValueError("Column '% of Peak Power' not found in load profile CSV")
        
        # Convert percentage to actual kW values
        df['Load (kW)'] = df['% of Peak Power'] * peak_power_kw / 100
        
        # Create datetime index (assuming half-hourly data starting from beginning of year)
        # You may need to adjust this based on your actual data structure
        start_date = pd.Timestamp('2024-01-01')
        df['Datetime'] = pd.date_range(start=start_date, periods=len(df), freq='30min')
        df.set_index('Datetime', inplace=True)
        
        self.load_profile_df = df
        
        print(f"Loaded {len(df)} load profile records")
        print(f"Load range: {df['Load (kW)'].min():.1f} to {df['Load (kW)'].max():.1f} kW")
        
        return df
    
    def prepare_hourly_data(self, hours: int = 24, 
                           start_datetime: str = None) -> Tuple[List[float], List[float]]:
        """
        Prepare hourly data for optimization
        
        Args:
            hours: Number of hours to prepare
            start_datetime: Starting datetime (optional)
            
        Returns:
            Tuple of (wholesale_prices, load_values) lists
        """
        if self.prices_df is None:
            raise ValueError("Prices data not loaded. Call load_wholesale_prices() first.")
        
        # Get subset of data
        if start_datetime:
            start_idx = pd.to_datetime(start_datetime)
            prices_subset = self.prices_df[self.prices_df.index >= start_idx].head(hours)
        else:
            prices_subset = self.prices_df.head(hours)
        
        # Extract prices
        wholesale_prices = prices_subset['Price ($/kWh)'].tolist()
        
        # Handle load data if available
        load_values = None
        if self.load_profile_df is not None:
            # Resample to hourly if needed (assuming half-hourly data)
            load_hourly = self.load_profile_df.resample('h').mean()
            
            # Align with price data timeframe
            common_idx = prices_subset.index.intersection(load_hourly.index)
            if len(common_idx) > 0:
                load_values = load_hourly.loc[common_idx, 'Load (kW)'].tolist()
        
        return wholesale_prices, load_values


def run_wholesale_optimization(prices_csv_path: str,
                             country: str = 'Germany',
                             hours: int = 24,
                             battery_capacity_kwh: float = 1000,
                             battery_power_kw: float = 250,
                             start_date: str = None,
                             save_results: bool = True):
    """
    Run wholesale arbitrage optimization with real price data
    
    Args:
        prices_csv_path: Path to wholesale prices CSV
        country: Country to analyze
        hours: Number of hours to optimize
        battery_capacity_kwh: Battery capacity in kWh
        battery_power_kw: Battery power rating in kW
        start_date: Start date for analysis (YYYY-MM-DD)
        save_results: Whether to save results to CSV
    """
    
    # Initialize data loader
    loader = BESSDataLoader(prices_csv_path)
    
    # Load price data
    prices_df = loader.load_wholesale_prices(country=country, start_date=start_date)
    
    # Prepare hourly data
    wholesale_prices, _ = loader.prepare_hourly_data(hours=hours)

    #Prepare BESS parameters through PyBAMM
    max_charge, max_discharge, eff_chg, eff_dis, max_soc, min_soc, daily_cycles = \
        simulate_battery_profile(battery_capacity_kwh, battery_power_kw, hours)
    
    # Create BESS parameters
    params = BESSParameters(
        total_capacity=battery_capacity_kwh,
        total_rated_power=battery_power_kw,
        rte=(sum(eff_chg) / len(eff_chg) + sum(eff_dis) / len(eff_dis)) / 2,
        initial_soc=0.5,          # Start at 50% SoC
        max_soc=max_soc,               # From PyBAMM
        min_soc=min_soc,               # From PyBAMM
        daily_cycles=daily_cycles,    # From PyBAMM
        replacement_cost=300      # $/kWh replacement cost
    )
    
    print(f"\n=== Running Wholesale Arbitrage Optimization ===")
    print(f"Country: {country}")
    print(f"Battery: {battery_capacity_kwh} kWh / {battery_power_kw} kW")
    print(f"Time period: {hours} hours starting from {prices_df.index[0]}")
    
    # Create and solve model
    from pulp import PULP_CBC_CMD, value
    
    wholesale_model = WholesaleOnlyArbitrage(params)
    model = wholesale_model.create_model(wholesale_prices)
    model.solve(PULP_CBC_CMD(msg=0))
    
    if model.status == 1:  # Optimal
        results = wholesale_model.extract_results(model, list(range(hours)))
        metrics = calculate_financial_metrics(results, params, 
                                            prices_wholesale=wholesale_prices)
        
        print(f"\n=== Optimization Results ===")
        print(f"Status: Optimal solution found")
        print(f"Objective value: ${value(model.objective):.2f}")
        print(f"Total charged: {metrics['E_chg']:.1f} kWh")
        print(f"Total discharged: {metrics['E_discharge']:.1f} kWh")
        print(f"Average daily cycles: {metrics['Avg_cycles']:.2f}")
        print(f"Captured spread: ${metrics.get('Spread_wholesale', 0):.4f}/kWh")
        
        # Calculate revenue
        revenue = sum([d * p for d, p in zip(results['discharge'], wholesale_prices)])
        cost = sum([c * p for c, p in zip(results['charge'], wholesale_prices)])
        net_profit = revenue - cost
        
        print(f"\n=== Financial Summary ===")
        print(f"Discharge revenue: ${revenue:.2f}")
        print(f"Charging cost: ${cost:.2f}")
        print(f"Net profit (before degradation): ${net_profit:.2f}")
        print(f"Daily profit rate: ${net_profit * 24 / hours:.2f}/day")
        
        if save_results:
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Datetime': prices_df.index[:hours],
                'Price ($/kWh)': wholesale_prices,
                'Charge (kWh)': results['charge'],
                'Discharge (kWh)': results['discharge'],
                'SoC (kWh)': results['soc'],
                'Net Power (kW)': [d - c for c, d in zip(results['charge'], results['discharge'])]
            })
            
            # Save to CSV
            output_filename = f"bess_results_{country}_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(output_filename, index=False)
            print(f"\nResults saved to: {output_filename}")
            
        return results, metrics
        
    else:
        print(f"Optimization failed with status: {model.status}")
        return None, None


# Example usage
if __name__ == "__main__":
    # Example: Run optimization for Germany for 168 hours (1 week)
    
    # You'll need to update these paths to match your file locations
    PRICES_CSV = "path/to/your/wholesale_prices.csv"
    LOAD_PROFILE_CSV = "path/to/your/industrial_load_profile.csv"  # Optional
    
    # Check if files exist
    if not os.path.exists(PRICES_CSV):
        print(f"Error: Prices file not found at {PRICES_CSV}")
        print("Please update the PRICES_CSV path in the script")
        sys.exit(1)
    
    # Run optimization
    results, metrics = run_wholesale_optimization(
        prices_csv_path=PRICES_CSV,
        country='Germany',
        hours=168,  # 1 week
        battery_capacity_kwh=1000,  # 1 MWh
        battery_power_kw=250,       # 250 kW (4-hour battery)
        start_date='2024-01-01',    # Adjust based on your data
        save_results=True
    )
    
    # You can also run for multiple countries
    countries_to_analyze = ['Germany', 'France', 'Spain', 'Netherlands']
    
    print("\n\n=== Multi-country Analysis ===")
    all_results = {}
    
    for country in countries_to_analyze:
        print(f"\n--- Analyzing {country} ---")
        try:
            results, metrics = run_wholesale_optimization(
                prices_csv_path=PRICES_CSV,
                country=country,
                hours=24,  # 1 day for quick comparison
                battery_capacity_kwh=1000,
                battery_power_kw=250,
                save_results=False
            )
            if metrics:
                all_results[country] = metrics
        except Exception as e:
            print(f"Error analyzing {country}: {e}")
    
    # Compare results
    if all_results:
        print("\n=== Country Comparison ===")
        print(f"{'Country':<15} {'Avg Cycles':<12} {'Spread ($/kWh)':<15}")
        print("-" * 45)
        for country, metrics in all_results.items():
            print(f"{country:<15} {metrics['Avg_cycles']:<12.2f} ${metrics.get('Spread_wholesale', 0):<14.4f}")
