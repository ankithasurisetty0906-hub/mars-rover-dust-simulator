import pandas as pd
import numpy as np

def calculate_solar_power(atm_row, dev_params, G0_val=590.0):
    """
    Calculates the instantaneous electrical power output of a solar panel
    under Martian dust storm conditions for a given time step.

    Parameters:
    atm_row (pd.Series): A row from the atmospheric data DataFrame, containing:
        - 'τ (optical depth)' (float): Atmospheric optical depth (unitless)
        - 'θ_z (deg)' (float): Solar zenith angle (degrees)
        - 'm_d (g/m²)' (float): Mass of dust per unit area on panel (g/m^2)
        - 'T_cell (°C)' (float): Cell operating temperature (Celsius)
    dev_params (dict): A dictionary containing static device parameters:
        - 'A' (float): Panel area (m^2)
        - 'eta_ref' (float): Module efficiency at reference temperature (0-1)
        - 'T_ref' (float): Reference temperature (Celsius)
        - 'alpha_eta' (float): Relative temperature coefficient (per Celsius)
        - 'kappa_s' (float): Extinction coefficient per mass for dust layer (m^2/g)
        - 'theta_i' (float): Panel tilt / incidence angle (degrees)
        - 'specific_power_to_weight' (float): Power required per unit of payload weight (W/kg)

    G0_val (float): Solar constant at Mars (W/m^2). Defaulted to 590 W/m^2.

    Returns:
    dict: A dictionary containing calculated power output and other metrics for this time step.
    """
    try:
        # Extract atmospheric parameters for this time step
        tau = atm_row['τ (optical depth)']
        theta_z = atm_row['θ_z (deg)']
        m_d = atm_row['m_d (g/m²)']
        T_cell = atm_row['T_cell (°C)']

        # Extract device parameters
        A = dev_params['A']
        eta_ref = dev_params['eta_ref']
        T_ref = dev_params['T_ref']
        alpha_eta = dev_params['alpha_eta']
        kappa_s = dev_params['kappa_s']
        theta_i = dev_params['theta_i']
        specific_power_to_weight = dev_params['specific_power_to_weight']

        # Convert angles from degrees to radians
        theta_z_rad = np.deg2rad(theta_z)
        theta_i_rad = np.deg2rad(theta_i)

        # Cosine terms
        mu_cos_theta_z = np.cos(theta_z_rad)
        cos_theta_i = np.cos(theta_i_rad)

        # Handle night-time or grazing angles (solar zenith angle >= 90 degrees)
        if mu_cos_theta_z <= 0.001: # Use a small epsilon for robustness
            power_output = 0.0
            effective_irradiance_on_panel = 0.0
            temp_efficiency_factor = 1.0 # Does not affect 0 power output
            soiling_transmittance = np.exp(-kappa_s * m_d) # Still compute for info
        else:
            # Attenuation exponent
            attenuation_exponent = tau / mu_cos_theta_z
            # Irradiance on surface projected by combined equation (incorporates cos_theta_i directly)
            effective_irradiance_on_panel = G0_val * cos_theta_i * np.exp(-attenuation_exponent)

            # Temperature efficiency factor
            temp_efficiency_factor = 1 + alpha_eta * (T_cell - T_ref)
            if temp_efficiency_factor < 0:
                temp_efficiency_factor = 0 # Efficiency cannot be negative

            # Soiling transmittance
            soiling_transmittance = np.exp(-kappa_s * m_d)
            if soiling_transmittance < 0:
                soiling_transmittance = 0 # Transmittance cannot be negative

            # Overall efficiency considering all factors (relative to eta_ref)
            overall_relative_eta = temp_efficiency_factor * soiling_transmittance

            # Final combined power output calculation
            power_output = A * eta_ref * effective_irradiance_on_panel * overall_relative_eta
            if power_output < 0: # Ensure power is not negative
                power_output = 0

        # Calculate estimated payload capacity
        estimated_payload_kg = power_output / specific_power_to_weight if specific_power_to_weight > 0 else 0

        return {
            "Power Output (W)": power_output,
            "Effective Irradiance on Panel (W/m^2)": effective_irradiance_on_panel,
            "Temperature Efficiency Factor": temp_efficiency_factor,
            "Soiling Transmittance": soiling_transmittance,
            "Overall Efficiency (%)": (eta_ref * temp_efficiency_factor * soiling_transmittance) * 100,
            "Estimated Payload Capacity (kg)": estimated_payload_kg
        }

    except KeyError as e:
        return {"Error": f"Missing expected column in atmospheric data: {e}. Please ensure Excel headers match: 'Time (h)', 'τ (optical depth)', 'θ_z (deg)', 'm_d (g/m²)', 'T_cell (°C)'"}
    except Exception as e:
        return {"Error": str(e)}

def get_device_parameters_manual():
    """Prompts the user for device parameters manually, with default values."""
    print("\n--- Enter Device/Rover Parameters ---")
    dev_params = {}
    try:
        # Using .get() with a default value allows empty input to use the default
        dev_params['A'] = float(input("Enter Panel Area (m^2) [1.0]: ") or 1.0)
        dev_params['eta_ref'] = float(input("Enter Reference efficiency η_ref (decimal 0-1) [0.2]: ") or 0.2)
        dev_params['alpha_eta'] = float(input("Enter Temperature coefficient α_η (per °C, typical -0.004) [-0.004]: ") or -0.004)
        dev_params['kappa_s'] = float(input("Enter Dust extinction κ_s (m²/g) [0.01]: ") or 0.01)
        dev_params['theta_i'] = float(input("Enter Panel tilt / incidence angle θ_i (deg) [30.0]: ") or 30.0)
        dev_params['T_ref'] = float(input("Enter Reference temperature T_ref (°C) [25.0]: ") or 25.0)

        # New/moved parameters
        dev_params['battery_capacity_Wh'] = float(input("Enter Battery capacity (Wh) [0 if N/A] [200.0]: ") or 200.0)
        dev_params['rover_power_requirement_W'] = float(input("Enter Rover continuous power requirement (W) [20.0]: ") or 20.0)
        dev_params['specific_power_to_weight'] = float(input("Enter Specific power-to-weight (W per kg payload) [10.0]: ") or 10.0)

    except ValueError:
        print("Invalid input. Please ensure all entries are numbers.")
        return None
    return dev_params

def main():
    excel_file = 'mars_dust_data.xlsx'
    G0_CONSTANT = 590.0 # Solar constant at Mars

    try:
        # Read atmospheric time-series data
        df_atm = pd.read_excel(excel_file)

        # Validate expected columns
        expected_atm_columns = ['Time (h)', 'τ (optical depth)', 'θ_z (deg)', 'm_d (g/m²)', 'T_cell (°C)']
        if not all(col in df_atm.columns for col in expected_atm_columns):
            missing_cols = [col for col in expected_atm_columns if col not in df_atm.columns]
            raise KeyError(f"Missing required columns in Atmosphere sheet: {missing_cols}. "
                           "Use headers like 'Time (h)', 'τ (optical depth)', 'θ_z (deg)', 'm_d (g/m²)', 'T_cell (°C)'.")

        print("--- Atmospheric Parameters from Excel (first few rows) ---")
        print(df_atm.head())

        # Get device parameters manually
        dev_params = get_device_parameters_manual()
        if dev_params is None:
            return # Exit if manual input was invalid

        print("\n--- Device/Rover Parameters (Manual Entry) ---")
        for key, value in dev_params.items():
            print(f"{key}: {value}")

        print("\n--- Running Simulation ---")

        simulation_results = []
        for index, row in df_atm.iterrows():
            time_h = row['Time (h)']
            results = calculate_solar_power(row, dev_params, G0_val=G0_CONSTANT)
            if "Error" in results:
                print(f"Error at Time {time_h}h: {results['Error']}")
                return # Stop if there's a critical error
            results['Time (h)'] = time_h # Add time to results
            simulation_results.append(results)

        df_results = pd.DataFrame(simulation_results)

        print("\n--- Simulation Summary ---")
        print(f"Total simulation duration: {df_results['Time (h)'].max()} hours")
        print(f"Minimum Power Output: {df_results['Power Output (W)'].min():.2f} W")
        print(f"Maximum Power Output: {df_results['Power Output (W)'].max():.2f} W")
        print(f"Average Power Output: {df_results['Power Output (W)'].mean():.2f} W")

        # Analyze blackout duration
        # Assuming blackout if power output is less than rover's continuous requirement
        rover_req = dev_params['rover_power_requirement_W']
        blackout_periods = df_results[df_results['Power Output (W)'] < rover_req]
        blackout_hours = len(blackout_periods) * (df_results['Time (h)'].diff().mean() or 1) # Assuming 1-hour intervals if diff is NaN
        print(f"Hours below rover's continuous power requirement ({rover_req}W): {blackout_hours:.2f} hours")

        # Battery analysis (simple cumulative model for now)
        battery_capacity = dev_params['battery_capacity_Wh']
        battery_charge_Wh = battery_capacity # Start fully charged
        battery_states = []

        for index, row in df_results.iterrows():
            generated_power_W = row['Power Output (W)']
            time_step_h = df_results['Time (h)'].diff().iloc[index] if index > 0 else 0 # Assuming first time step is 0 or 1h
            if index == 0: time_step_h = df_results['Time (h)'].iloc[0] if df_results['Time (h)'].iloc[0] > 0 else 1 # default to 1h if start at 0

            # Energy generated/consumed in this time step (Wh)
            energy_generated_Wh = generated_power_W * time_step_h
            energy_consumed_Wh = rover_req * time_step_h

            net_energy_change_Wh = energy_generated_Wh - energy_consumed_Wh
            battery_charge_Wh += net_energy_change_Wh
            battery_charge_Wh = min(battery_charge_Wh, battery_capacity) # Cap at max capacity
            battery_charge_Wh = max(battery_charge_Wh, 0) # Cannot go below zero

            battery_states.append(battery_charge_Wh)

        df_results['Battery Charge (Wh)'] = battery_states
        print(f"Minimum Battery Charge: {df_results['Battery Charge (Wh)'].min():.2f} Wh")
        print(f"Maximum Payload Capacity (overall min power): {df_results['Estimated Payload Capacity (kg)'].min():.2f} kg (based on lowest power output)")

        # You can save this df_results to a CSV or Excel for further analysis/plotting
        df_results.to_csv('simulation_output.csv', index=False)
        print("\nFull simulation results saved to 'simulation_output.csv'")
        print("You can now use this CSV to create plots and visualize the data.")

    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found. Please create it with atmospheric data.")
    except KeyError as e:
        print(f"Error during computation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()