import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import newton
import plotly.graph_objects as go

# ---------------------------------------------------------------------------- #
# Sellmeier + Thermo-Optic (Thorlabs model) for Type-0 PPKTP
# ---------------------------------------------------------------------------- #

def sellmeier(w, pol):
    if pol == 'z':
        return np.sqrt(np.abs(
            2.12725 + (1.18431 / (w**2 - 0.0514852) + 0.6603 / (w**2 - 100.00507) - 9.68956e-3) * (w**2)
        ))
    else:
        return np.sqrt(np.abs(
            2.09930 + (0.922683 / (w**2 - 0.0467695) - 0.0138404) * (w**2)
        ))

def temperature_dependence(w, pol):
    if pol == "z":
        return (1e-6 * (4.1010 * w**-3 - 8.9603 * w**-2 + 9.9228 * w**-1 + 9.9587) +
                1e-8 * (3.1481 * w**-3 - 9.8136 * w**-2 + 10.459 * w**-1 - 1.1882))
    else:
        return (1e-6 * (2.6486 * w**-3 - 6.0629 * w**-2 + 6.3061 * w**-1 + 6.2897) +
                1e-8 * (1.3470 * w**-3 - 3.5770 * w**-2 + 2.2244 * w**-1 - 0.14445))

def n(w, T, pol, T_ref=25):
    return sellmeier(w, pol) + temperature_dependence(w, pol) * (T - T_ref)

def poling_period(w1, w2, w3, T, T_ref=25):
    return 1 / (n(w3, T, "z", T_ref) / w3 - n(w2, T, "z", T_ref) / w2 - n(w1, T, "z", T_ref) / w1)

def solve_w1_for_period(target_period, w3, T, T_ref=25):
    def equation(w1):
        w2 = 1 / (1 / w3 - 1 / w1)
        return poling_period(w1, w2, w3, T, T_ref) - target_period

    w1_guess = 1 / (1 / w3 - 1 / 0.9)
    return newton(equation, w1_guess)

# ---------------------------------------------------------------------------- #
# Streamlit App
# ---------------------------------------------------------------------------- #

def run():
    

    # Sidebar Inputs
    st.sidebar.header("Simulation Parameters")
    decimals = st.sidebar.slider("Decimal places", 0, 10, 4)
    w3 = st.sidebar.number_input("Pump Wavelength λp (µm)", 0.3, 1.0, 0.405, 0.001, format=f"%.{decimals}f")
    w1_example = st.sidebar.number_input("Example Signal Wavelength λi (µm)", 0.7, 1.2, 0.81, 0.001, format=f"%.{decimals}f")

    T0 = st.sidebar.number_input("Operating Temp T₀ (°C)", 0.000, 70.000, 25.000, 1.000, format=f"%.{decimals}f")
    T_ref = st.sidebar.number_input("Reference Temp T_ref (°C)", 0.000, 150.000, 25.000, 1.000, format=f"%.{decimals}f")

    # 🔴 Check condition: T₀ > T_ref
    if T0 < T_ref:
        st.sidebar.error("Operating Temperature T₀ must be greater than or equal to Reference Temperature T_ref.")
        return

    auto_calc = st.sidebar.checkbox("Auto-calculate Λ at T₀", value=True)

    if auto_calc:
        w2_example = 1 / (1 / w3 - 1 / w1_example)
        Λ_fixed = poling_period(w1_example, w2_example, w3, T0, T_ref)
    else:
        Λ_fixed = st.sidebar.number_input("Poling Period Λ (µm)", 3.0000, 4.0000, 3.4250, 0.0001, format=f"%.{decimals}f")

    T_min = st.sidebar.number_input("Min Temp (°C)", 0.000, 100.000, 25.000, 1.000)
    T_max = st.sidebar.number_input("Max Temp (°C)", 25.000, 150.000, 75.000, 1.000)
    points = st.sidebar.slider("Temperature Points", 10, 500, 51)

    if T_max <= T_min:
        st.sidebar.error("T_max must be greater than T_min.")
        return

    # Compute tuning data
    temps = np.linspace(T_min, T_max, points)
    idlers = []
    signals = []

    for T in temps:
        try:
            w1 = solve_w1_for_period(Λ_fixed, w3, T, T_ref)
            w2 = 1 / (1 / w3 - 1 / w1)
            idlers.append(w1 * 1000)  # nm
            signals.append(w2 * 1000)
        except RuntimeError:
            idlers.append(np.nan)
            signals.append(np.nan)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temps, y=signals, mode='lines+markers', name='Signal λs [nm]',
        hovertemplate=f'T = %{{x:.2f}} °C<br>λs = %{{y:.{decimals}f}} nm'
    ))
    fig.add_trace(go.Scatter(
        x=temps, y=idlers, mode='lines+markers', name='Idler λi [nm]',
        hovertemplate=f'T = %{{x:.2f}} °C<br>λi = %{{y:.{decimals}f}} nm'
    ))

    fig.update_layout(
        title=f'Type-0 SPDC Tuning Curve (λp = {w3:.{decimals}f} µm, Λ = {Λ_fixed:.{decimals}f} µm @ T₀ = {T0:.{decimals}f} °C)',
        xaxis_title='Temperature [°C]',
        yaxis_title='Wavelength [nm]',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run()


