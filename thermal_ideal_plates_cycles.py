#!/usr/bin/env python
# python 3
# pylint: disable=
##    @file:    thermal_play.py
#     @name:    "Aaron Snyder"
#     @date:    24 Aug 2023
####################################################################################################
# This script models a 1D-Thermal-RC system using Ordinary Differential Equations (ODEs) with a 
# basic heat limited ramp. The governing equations are solved with numerical methods
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#------------------------------------ Constants -------------------------------------
# Material properties - book values 
rho_pp = 900    # Density of pp in kg/m^3
rho_al = 2700   # Density of aluminum in kg/m^3
rho_w = 1000    # Density of water in kg/m^3

c_pp = 1920     # Specific heat of pp in J/kg.K
c_al = 897      # Specific heat of aluminum in J/kg.K
c_w = 4186      # Specific heat of water in J/kg.K

k_pp = 0.28     # Thermal conductivity of pp in W/m.K
k_al = 237      # Thermal conductivity of aluminum in W/m.K

#------------------------------------ Properties ------------------------------------
# Geometric properties - tunable parameters
L_pp = 7.6e-5   # Thickness of pp prism in m (0.076mm)
L_al_heating = 0.005  # Thickness of aluminum block during heating in m (5mm)
L_al_cooling = 0.1  # Increased thickness of aluminum block during cooling in m

A_pp = 1.26e-5  # Area of pp prism in m^2 (Marc to provide XY dim)
A_al = A_pp * 8     # Area of aluminum block in m^2 (Marc to provide XY dim)

V_w = 1.5e-8    # Volume of water in m^3 (15uL)

#------------------------------------ Derived Values --------------------------------
# Thermal capacitance - calculated
C_pp = rho_pp * L_pp * A_pp * c_pp        # Thermal capacitance of pp
C_w = rho_w * V_w * c_w                   # Thermal capacitance of water
C_al_heating = rho_al * L_al_heating * A_al * c_al  # Thermal capacitance of aluminum during heating
C_al_cooling = rho_al * L_al_cooling * A_al * c_al  # Increased thermal capacitance of aluminum during cooling

# Thermal resistance
R_al_pp = L_al_heating / (k_al * A_al)            # Thermal resistance between aluminum and pp
R_pp_w = L_pp / (k_pp * A_pp)             # Thermal resistance between pp and water

P_input = 8.33  # Heater power in watts (value for individual well)
cooling = False
#------------------------------------ ODE Solver -----------------------------------
# Apply ODE
def thermal_model(t, T):
    T_al, T_pp, T_w = T
    
    if cooling:
        Q_heater_al = 0
    else:
        Q_heater_al = P_input if T_al < 110 else 0
    
    Q_al_pp = (T_al - T_pp) / R_al_pp

    dT_al_dt = (Q_heater_al - Q_al_pp) / (C_al_heating if not cooling else C_al_cooling)
    dT_pp_dt = (Q_al_pp - (T_pp - T_w) / R_pp_w) / C_pp
    dT_w_dt = (T_pp - T_w) / (R_pp_w * C_w)

    return [dT_al_dt, dT_pp_dt, dT_w_dt]

#------------------------------------ Event Functions -------------------------------
def event_heat_on(t, T):
    return T[2] - 100
event_heat_on.terminal = True

def event_cool_end(t, T):
    return T[2] - 60
event_cool_end.terminal = True
#------------------------------------ Solve for Values -----------------------------
# Initialize empty arrays to store results for 40 cycles
t_cycles = np.array([])
T_al_cycles = np.array([])
T_pp_cycles = np.array([])
T_w_cycles = np.array([])

# Initialize the first set of initial conditions
T0 = [100, 60, 60]
cool_al_final_temp = 50  # Initial cooling aluminum plate temperature
t_start = 0

# Loop through 40 cycles
for cycle in range(40):
    cooling = False  # Reset the cooling flag
    sol1 = solve_ivp(thermal_model, [t_start, t_start+2000], T0, events=event_heat_on, t_eval=np.linspace(t_start, t_start+2000, 4000))
    cooling = True  # Begin cooling process
    
    T0_cool = [cool_al_final_temp, sol1.y[1, -1], sol1.y[2, -1]]  # Use the last cooling temperature for aluminum
    sol2 = solve_ivp(thermal_model, [sol1.t[-1], sol1.t[-1]+2000], T0_cool, events=event_cool_end, t_eval=np.linspace(sol1.t[-1], sol1.t[-1]+2000, 4000))

    # Concatenate time and temperature arrays
    t_cycles = np.concatenate((t_cycles, sol1.t, sol2.t))
    T_al_cycles = np.concatenate((T_al_cycles, sol1.y[0], sol2.y[0]))
    T_pp_cycles = np.concatenate((T_pp_cycles, sol1.y[1], sol2.y[1]))
    T_w_cycles = np.concatenate((T_w_cycles, sol1.y[2], sol2.y[2]))

    # Update the initial conditions and the starting time for the next cycle
    T0 = [sol2.y[0, -1], sol2.y[1, -1], sol2.y[2, -1]]
    cool_al_final_temp = sol2.y[0, -1]  # Update the cooling aluminum plate temperature
    t_start = sol2.t[-1]

# Plotting
plt.plot(t_cycles, T_al_cycles, label='Aluminum Temperature')
plt.plot(t_cycles, T_pp_cycles, label='Polypropylene Temperature')
plt.plot(t_cycles, T_w_cycles, label='Water Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.title('Temperature Evolution over 40 Cycles')
plt.show()

# Print the results
print(f"Time for water to reach 100°C in the last cycle: {sol1.t[-1] - t_start:.2f} seconds")
print(f"Time taken for the aluminum to cool back down to 50°C in the last cycle: {sol2.t[-1] - sol1.t[-1]:.2f} seconds")

# Calculate and print the total time for 40 cycles
total_time = t_cycles[-1] - t_cycles[0]
print(f"Total time for 40 cycles: {total_time/60:.2f} minutes")