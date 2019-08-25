# Phyton CODE used in my Summer School Project 2019:

Aim is to explore new methods of fast ion beam spectroscopy

The core feature of this is solving the two level system (toy model) coupled to a EM mode of variable strength and frequency. The core equations are Maxwell-Bloch equations. Precise implementation varies from script to script.

- BONUS feature is informing the user when the simulation starts and finishes (with results) via email. Implemented on couple of scripts (the ones which have imported send_email package)

The Scripts are:

- two_level_complete_header:
  - have Bloch equations (von_neumann* functions) for different configurations
  - helper functions like pulsing and ramping used in both frequency definitions (detunning variables) and power definitions (rabi_freq* variables)
  - comsol_doppler_shift which can interpolate the functions form from given data in the form of .txt file
