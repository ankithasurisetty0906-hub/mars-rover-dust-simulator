import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="Mars Rover Dust Simulator")


# --- Core Calculation Function (unchanged) ---
def calculate_solar_power(atm_row, dev_params, G0_val=590.0):
    """
    Calculates the instantaneous electrical power output of a solar panel
    under Martian dust storm conditions for a given time step.
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

# --- Streamlit App ---
def app():
    

    st.title("Mars Rover Solar Power Simulator")
    st.markdown("Simulate the power output of a Mars rover's solar panel under varying atmospheric and dust conditions.")

    excel_file = 'mars_dust_data.xlsx'
    G0_CONSTANT = 590.0 # Solar constant at Mars

    # --- Sidebar for Device/Rover Parameters ---
    st.sidebar.header("Device/Rover Parameters")

    dev_params = {}
    dev_params['A'] = st.sidebar.number_input("Panel Area (m²)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    dev_params['eta_ref'] = st.sidebar.slider("Reference efficiency η_ref (0-1)", value=0.2, min_value=0.01, max_value=0.5, step=0.01)
    dev_params['alpha_eta'] = st.sidebar.number_input("Temperature coefficient α_η (per °C)", value=-0.004, min_value=-0.01, max_value=0.0, step=0.001, format="%.4f")
    dev_params['kappa_s'] = st.sidebar.number_input("Dust extinction κ_s (m²/g)", value=0.01, min_value=0.0, max_value=0.1, step=0.001, format="%.3f")
    dev_params['theta_i'] = st.sidebar.slider("Panel tilt / incidence angle θ_i (deg)", value=30.0, min_value=0.0, max_value=90.0, step=1.0)
    dev_params['T_ref'] = st.sidebar.number_input("Reference temperature T_ref (°C)", value=25.0, min_value=-50.0, max_value=50.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Rover System Details")
    dev_params['battery_capacity_Wh'] = st.sidebar.number_input("Battery capacity (Wh)", value=200.0, min_value=0.0, step=10.0)
    dev_params['rover_power_requirement_W'] = st.sidebar.number_input("Rover continuous power requirement (W)", value=20.0, min_value=0.0, step=1.0)
    dev_params['specific_power_to_weight'] = st.sidebar.number_input("Specific power-to-weight (W per kg payload)", value=10.0, min_value=1.0, step=1.0)


    # --- Main Area ---
    st.subheader("1. Atmospheric Data Input")
    st.info(f"Expecting an Excel file named `{excel_file}` in the same directory as this script.")

    try:
        # Read atmospheric time-series data
        df_atm = pd.read_excel(excel_file)

        # Validate expected columns
        expected_atm_columns = ['Time (h)', 'τ (optical depth)', 'θ_z (deg)', 'm_d (g/m²)', 'T_cell (°C)']
        if not all(col in df_atm.columns for col in expected_atm_columns):
            missing_cols = [col for col in expected_atm_columns if col not in df_atm.columns]
            st.error(f"Missing required columns in '{excel_file}': {missing_cols}. "
                           "Please ensure headers match: 'Time (h)', 'τ (optical depth)', 'θ_z (deg)', 'm_d (g/m²)', 'T_cell (°C)'.")
            st.stop() # Stop execution if data is malformed

        st.markdown("--- Atmospheric Parameters from Excel (first few rows) ---")
        st.dataframe(df_atm.head())

        st.subheader("2. Run Simulation")
        if st.button("Run Simulation"):
            st.markdown("---")
            st.subheader("3. Simulation Results")

            simulation_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for index, row in df_atm.iterrows():
                time_h = row['Time (h)']
                results = calculate_solar_power(row, dev_params, G0_val=G0_CONSTANT)
                if "Error" in results:
                    st.error(f"Error at Time {time_h}h: {results['Error']}")
                    st.stop() # Stop if there's a critical error
                results['Time (h)'] = time_h # Add time to results
                simulation_results.append(results)

                progress = (index + 1) / len(df_atm)
                progress_bar.progress(progress)
                status_text.text(f"Calculating... {int(progress * 100)}%")

            status_text.success("Simulation complete!")
            progress_bar.empty() # Clear the progress bar


            df_results = pd.DataFrame(simulation_results)

            # --- Simulation Summary ---
            st.markdown("### Simulation Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Duration", f"{df_results['Time (h)'].max():.0f} hours")
            col2.metric("Min Power Output", f"{df_results['Power Output (W)'].min():.2f} W")
            col3.metric("Max Power Output", f"{df_results['Power Output (W)'].max():.2f} W")
            col4.metric("Average Power Output", f"{df_results['Power Output (W)'].mean():.2f} W")

            # --- Power Output Chart ---
            st.markdown("### Power Output Over Time")
            st.line_chart(df_results[['Time (h)', 'Power Output (W)']].set_index('Time (h)'))

            # --- Soiling and Efficiency Chart ---
            st.markdown("### Soiling and Efficiency Factors")
            st.line_chart(df_results[['Time (h)', 'Soiling Transmittance', 'Temperature Efficiency Factor']].set_index('Time (h)'))

            # --- Blackout Analysis ---
            rover_req = dev_params['rover_power_requirement_W']
            blackout_periods = df_results[df_results['Power Output (W)'] < rover_req]
            # Assuming 1-hour intervals if diff is NaN, otherwise use average difference
            time_step_duration = df_results['Time (h)'].diff().mean() if len(df_results) > 1 else 1
            blackout_hours = len(blackout_periods) * time_step_duration
            st.metric(f"Hours below rover's continuous power requirement ({rover_req}W)", f"{blackout_hours:.2f} hours")


            # --- Battery Analysis ---
            st.markdown("### Battery Charge Over Time")
            battery_capacity = dev_params['battery_capacity_Wh']
            battery_charge_Wh = battery_capacity # Start fully charged
            battery_states = []

            for index, row in df_results.iterrows():
                generated_power_W = row['Power Output (W)']
                # Use actual time step duration from data or default to 1h
                time_step_h = df_results['Time (h)'].diff().iloc[index] if index > 0 else (df_results['Time (h)'].iloc[0] if df_results['Time (h)'].iloc[0] > 0 else 1)

                energy_generated_Wh = generated_power_W * time_step_h
                energy_consumed_Wh = rover_req * time_step_h

                net_energy_change_Wh = energy_generated_Wh - energy_consumed_Wh
                battery_charge_Wh += net_energy_change_Wh
                battery_charge_Wh = min(battery_charge_Wh, battery_capacity) # Cap at max capacity
                battery_charge_Wh = max(battery_charge_Wh, 0.0) # Cannot go below zero

                battery_states.append(battery_charge_Wh)

            df_results['Battery Charge (Wh)'] = battery_states
            st.line_chart(df_results[['Time (h)', 'Battery Charge (Wh)']].set_index('Time (h)'))

            st.metric("Minimum Battery Charge", f"{df_results['Battery Charge (Wh)'].min():.2f} Wh")
            st.metric("Maximum Payload Capacity (based on lowest power output)", f"{df_results['Estimated Payload Capacity (kg)'].min():.2f} kg")

            # --- Raw Data and Download ---
            st.markdown("### Detailed Simulation Data")
            st.dataframe(df_results)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Simulation Results (CSV)",
                data=csv,
                file_name='simulation_output.csv',
                mime='text/csv',
            )
            st.success("Full simulation results are available for download above.")

    except FileNotFoundError:
        st.error(f"Error: The Excel file '{excel_file}' was not found. Please ensure 'mars_dust_data.xlsx' is in the same directory as this script.")
        st.markdown("You need to create an Excel file named `mars_dust_data.xlsx` with columns: `Time (h)`, `τ (optical depth)`, `θ_z (deg)`, `m_d (g/m²)`, `T_cell (°C)`.")
        st.markdown("Here's an example of what the first few rows of your Excel file should look like:")
        example_data = {
            'Time (h)': [0, 1, 2, 3, 4],
            'τ (optical depth)': [0.5, 1.5, 3.0, 2.5, 1.0],
            'θ_z (deg)': [45, 30, 30, 45, 60],
            'm_d (g/m²)': [5, 20, 60, 70, 50],
            'T_cell (°C)': [5, 0, -5, -10, -15]
        }
        st.dataframe(pd.DataFrame(example_data))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    app()