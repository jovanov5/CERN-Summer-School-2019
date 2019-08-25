# Phyton CODE used in my Summer School Project 2019:

Aim is to explore new methods of fast ion beam spectroscopy

The core feature of this is solving the two level system (toy model) coupled to a EM mode of variable strength and frequency. The core equations are Maxwell-Bloch equations. Precise implementation varies from script to script.

- BONUS feature is informing the user when the simulation starts and finishes (with results) via email. Implemented on couple of scripts (the ones which have imported send_email package)

The Scripts are:

- two_level_complete_header:
  - has Bloch equations (von_neumann* functions) for different configurations
  - helper functions like pulsing and ramping used in both frequency definitions (detunning variables) and power definitions (rabi_freq* variables)
  - comsol_doppler_shift which can interpolate the functions form from given data in the form of .txt file
- send_email:
  - has two emailing functions send_email to send an email to the user when it's finished
  - send_start to email the user when it started
  - both having capability to send a number of attacjments (.dfs, .py, .npy, .txt)
- all other scripts:
  -tailor-made for emulating given experimental setups or to explore and optimize other
- two miscellanies scripts:
  - MATLAB_like_init
  - script_1
- Mathematica/ code for analytical treatment of doppler switching 
- COSMOL/ data and constructions
- plots/
