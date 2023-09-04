import matplotlib.pyplot as plt 
import numpy as np 

# Thermal resistance of the thermal paste, aluminum block and thermal sensor 
R_p = .05 # heater resistor to thermal paste
R_a = .01 # thermal paste to aluminum heater block
R_s = .03  # heater block to sensor

# Thermal capacitance of the heater resistor, thermal paste, aluminum block and thermal sensor 
C_r = 0.00001

C_paste = 0.0001

m_a = 0.005
c_a = 900 
rho_a = 2700
C_alum = m_a * c_a * rho_a

m_s = 0.01 
c_s = 800 
rho_s = .005
C_sens = m_s * c_s * rho_s

C_th = C_r + C_paste + C_alum + C_sens 
R_th = R_p + R_a + R_s 

# Heat transfer coefficient
h = 15 # W/m^2·K
A = 0.0018952 # m^2
R_conv = 1/(h*A)

R_total = R_th + R_conv

tau = R_total * C_th

# Power input to the heating resistor 
P = 11.2 * 9

# Time constant 
t = np.linspace(0, 4 * tau, 1000) 

# Initia temperature of sensor
Ti = 20

# Final temperature of the sensor
Tf = Ti + P/(C_sens)/R_total
# Tf = 100

# Temperature rise of the sensor 
T = Ti + (Tf - Ti) * (1 - np.exp(-t / tau))  

# Plot the temperature rise of the sensor vs time 
plt.plot(t, T) 
plt.xlabel('Time (s)') 
plt.ylabel('Temperature (C)') 
plt.title('Temperature Rise of the Sensor vs Time') 
plt.show()
