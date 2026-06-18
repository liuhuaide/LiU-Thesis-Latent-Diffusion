"""
Constants for the data assimilation project.
"""
nx, ny = 64, 64  # Spatial dimensions of the data
data_mean = 0.0  # Mean of the data
data_std = 2660  # Standard deviation of the data
# Scaling factor for the data, needed to convert the units from the numerical model that propagete the states
scalefact = 0.003061224412462883
