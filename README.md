DiffusionFitting
================

DiffusionFitting is a Python-based program for fitting diffusion profiles using a numerical solution to Fick’s second law based on finite difference method. It supports concentration-dependent diffusion and automatically optimizes key physical parameters such as D, C0, and Cbgr using non-linear least squares. 

<img width="1017" height="681" alt="image" src="https://github.com/user-attachments/assets/a2219dd9-6026-4bba-8e27-dcd60121e81c" />



Features
--------

- Fits experimental diffusion data with flexible numerical simulations.
- Supports concentration-dependent diffusion (parameter n).
- Optimizes diffusion parameters automatically using non-linear least squares.
- Graphical User Interface (GUI) built with Tkinter.
- Advanced settings for fine-tuning initial guesses and parameter boundaries.
- Export simulation results and plots.

Installation
------------

Option 1: Run from Python source

1. Clone the repository:
   git clone https://github.com/Arianna-Projects/DiffusionFitting.git
   cd DiffusionFitting

2. Install required Python packages:
   pip install numpy scipy matplotlib

3. Run the program:
   python FitDiffusionODEToData_Final.py

Option 2: Download Executable

The latest Windows executable & accompanying folder can be downloaded from the Releases page:

Download DiffusionFitting v1.0:
https://github.com/Arianna-Projects/DiffusionFitting/releases/download/v1.0/FitDiffusionODEToData_Final.zip

Usage
-----

- Open the program and load your diffusion profile data (.txt file with two tab-separated columns: depth [µm] and concentration [1/cm3]).
- Adjust simulation settings as needed (n, Nx_sim, Nx_plot).
- Click "Fit" to optimize parameters.
- Save results and plots using the built-in buttons.

License
-------

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the LICENSE file for details.

Third-Party Libraries
---------------------

This project uses the following libraries, which remain under their respective licenses:

- SciPy: BSD 3-Clause License — https://scipy.org
- NumPy: BSD 3-Clause License — https://numpy.org
- Matplotlib: Matplotlib License — https://matplotlib.org
- Tkinter: Python Software Foundation License — https://www.python.org

Author
------

Arianna Jaroszynska — originally written in 2023, GUI added in 2025.  
GitHub: https://github.com/Arianna-Projects/DiffusionFitting

Thank you for using DiffusionFitting! :)
