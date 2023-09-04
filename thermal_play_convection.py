#!/usr/bin/env python
# python 3
# pylint: disable=
##    @file:    thermal_play.py
#     @name:    "Aaron Snyder"
#     @date:    24 Aug 2023
####################################################################################################
# This script models a 1D-Thermal-RC system using Ordinary Differential Equations (ODEs) with a 
# basic PI controlled heat limited ramp. The governing equations are solved with numerical methods
####################################################################################################
#%%
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
L_al = 0.005    # Thickness of aluminum block in m (5mm)

A_pp = 1.26e-5  # Area of pp prism in m^2 (Marc to provide XY dim)
A_al = A_pp     # Area of aluminum block in m^2 (Marc to provide XY dim)

V_w = 1.5e-8    # Volume of water in m^3 (15uL)

#------------------------------------ Derived Values --------------------------------
# Thermal capacitance - calculated
C_pp = rho_pp * L_pp * A_pp * c_pp        # Thermal capacitance of pp
C_w = rho_w * V_w * c_w                   # Thermal capacitance of water
C_al = rho_al * L_al * A_al * c_al        # Thermal capacitance of aluminum

# Thermal resistance
R_al_pp = L_al / (k_al * A_al)            # Thermal resistance between aluminum and pp
R_pp_w = L_pp / (k_pp * A_pp)             # Thermal resistance between pp and water

P_input = 8.33  # Heater power in watts (value for individual well)
#------------------------------------ Air Cooling -----------------------------------
# Air cooling properties - tunable parameters
air_velocity = 5                          # Air velocity in m/s
fin_pitch = 0.002                         # Fin pitch in meters
# total_fin_area = 0.000156               # Total area of all fins in m^2
# print(total_fin_area)
total_fin_area = A_pp * 4
air_temperature = 25                      # Air temperature in Celsius

# Constants
Pr = 0.7                                  # Prandtl number for air (kinematic viscosity/thermal diffusivity)
k_air = 0.0257                            # Thermal conductivity of air in W/(m*C)

# Calculate convective heat transfer coefficient
'''
air density value could be more sophisticated with a lookup table that guess the air
temperature near the reaction and uses the table to inject a new value for density
'''
air_density = 1.225                       # kg/m^2
dynamic_viscosity = 1.789e-5              # kg/(m*s)
Reynolds_number = (air_density * air_velocity * fin_pitch) / dynamic_viscosity
Nusselt_number = 0.023 * Reynolds_number**0.8 * Pr**0.3
convective_coefficient = (Nusselt_number * k_air) / fin_pitch

#------------------------------------ PI Controller --------------------------------
Kp = 0.01            # Proportional gain, adjust if needed
Ki = 0.001           # Integral gain, adjust if needed
int_error = 0        # Integral error initialization

#------------------------------------ Event Functions -------------------------------
# Set a maximum temperature for the aluminum
def event_al_max_temp(t, T):
    return T[0] - 110
event_al_max_temp.terminal = True

cooling = False  # Flag to check if cooling is activated

#------------------------------------ ODE Solver -----------------------------------
# Apply ODE
def thermal_model(t, T):
    global int_error
    T_al, T_pp, T_w = T
    
    error = 100 - T_w
    
    '''
    checks if both the current error and the integral error have the same sign 
    (either both positive or both negative). If they do, it implies that the 
    system is already saturated, and further accumulation of the integral error 
    should be prevented. In other words, this condition is checking if the 
    controller output is at its limits (e.g., the heater is fully on or off).
    '''
    # Anti-windup: Don't accumulate error if heater is fully on or off
    if not ((error > 0 and int_error > 0) or (error < 0 and int_error < 0)):
        int_error += error

    control_factor = Kp * error + Ki * int_error
    control_factor = max(0, min(control_factor, 1))

    if T_al > 108:  # Starting from 108°C to provide a margin
        control_factor *= (110 - T_al) / 2
    
    current_power = P_input * control_factor if not cooling else 0

    Q_heater_al = current_power
    Q_al_pp = (T_al - T_pp) / R_al_pp
    Q_pp_w = (T_pp - T_w) / R_pp_w

    if cooling:
        Q_cool_al = convective_coefficient * total_fin_area * (T_al - air_temperature)
        Q_cool_pp = convective_coefficient * A_pp * (T_pp - air_temperature)
    else:
        Q_cool_al = 0
        Q_cool_pp = 0

    dT_al_dt = (Q_heater_al - Q_al_pp - Q_cool_al) / C_al
    dT_pp_dt = (Q_al_pp - Q_pp_w - Q_cool_pp) / C_pp
    dT_w_dt = Q_pp_w / C_w

    return [dT_al_dt, dT_pp_dt, dT_w_dt]

#------------------------------------ Event Functions -------------------------------
def event_heat_off(t, T):
    return T[2] - 100
event_heat_off.terminal = True

def event_cool_end(t, T):
    return T[2] - 60
event_cool_end.terminal = True

#------------------------------------ Solve for Values -----------------------------
# Initial conditions ()
T0 = [100, 60, 60]

# Solve for heating phase
sol1 = solve_ivp(thermal_model, [0, 2000], T0, events=event_heat_off, t_eval=np.linspace(0, 2000, 4000))

cooling = True  # Begin cooling process

# Solve for cooling phase
T0_cool = [60, sol1.y[1, -1], sol1.y[2, -1]]
sol2 = solve_ivp(thermal_model, [sol1.t[-1], 4000], T0_cool, events=event_cool_end, t_eval=np.linspace(sol1.t[-1], 4000, 4000))

# Concatenate time and temperature arrays
t_total = np.concatenate((sol1.t, sol2.t))
T_total = np.column_stack((np.concatenate((sol1.y[0], sol2.y[0])),
                           np.concatenate((sol1.y[1], sol2.y[1])),
                           np.concatenate((sol1.y[2], sol2.y[2])))).T

#------------------------------------ Plot Results ---------------------------------
plt.plot(t_total, T_total[0], label='Aluminum Temperature')
plt.plot(t_total, T_total[1], label='Polypropylene Temperature')
plt.plot(t_total, T_total[2], label='Water Temperature')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.legend() 
plt.grid(True)
plt.title('Temperature Evolution with Power Input and Cooling')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'darkgray'
plt.rcParams['text.color'] = 'black'
plt.show()

#------------------------------------ Print Results --------------------------------
print(f"Time for water to reach 100°C: {sol1.t[-1]:.2f} seconds")
print(f"Time taken for the water to cool back down to 60°C after heating was turned off: {sol2.t[-1]-sol1.t[-1]:.2f} seconds")
print(f"Time for 40 cycles: {((sol1.t[-1] + sol2.t[-1]-sol1.t[-1]) * 40) / 60:.2f} minutes")
#%%