# FitDiffusionODEToData_Final.py v1.1
# Copyright (c) 2023-2025 Arianna Jaroszynska
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# This project uses third-party libraries:

#- SciPy (BSD 3-Clause License): https://scipy.org
#- NumPy (BSD 3-Clause License): https://numpy.org
#- Matplotlib (Matplotlib License): https://matplotlib.org
#- Tkinter (Python Software Foundation License): https://www.python.org

#These libraries remain under their respective licenses.

# Github repository for this project is in the link below
# https://github.com/Arianna-Projects/DiffusionFitting

#!/usr/bin/env python3
"""
Standalone GUI for fitting diffusion PDE to experimental data and plotting results.
Save this file and run with: python FitDiffusionODEToData.py
Dependencies: numpy, matplotlib, scipy, lmfit, tkinter (included with standard python installs)
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from lmfit import Parameters, minimize
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")

# --- Unit conversion ---
umtocm = 1e-4

# Default values
DEFAULT_NX_SIM = 200 # 200 sim points
DEFAULT_N = 0  # no concentration dependence
DEFAULT_TF = 30  # 30 min

# --- Utilities: vectorized RHS ---

def diffusion_rhs_vec(u, D, dx, n, C0):
    u_n = u**n
    dU = np.zeros_like(u)
    dU[1:-1] = (D / (C0**n) / (2*dx**2)) * (
        (u_n[1:-1] + u_n[2:]) * (u[2:] - u[1:-1]) -
        (u_n[:-2] + u_n[1:-1]) * (u[1:-1] - u[:-2])
    )
    return dU

# --- PDE solver ---
def solve_diffusion(t_annealing, D, Cbgr, C0, x_sim, n):
    dx = x_sim[1] - x_sim[0]
    u0 = np.full_like(x_sim, Cbgr)
    u0[0] = C0
    rhs = lambda t, u: diffusion_rhs_vec(u, D, dx, n, C0)
    sol = solve_ivp(rhs, [0, t_annealing], u0, method="BDF",
                    atol=1e-7, rtol=1e-5, t_eval=[t_annealing])
    return sol.y[:, -1]

# --- Residual (wrapper for lmfit) ---
def make_residual(x_fit, y_fit, Nx_sim, tf, n):
    def residual(paras):
        D = 10**paras['logD'].value
        C0 = 10**paras['logC0'].value
        Cbgr = 10**paras['logCbgr'].value

        x_sim = np.linspace(x_fit.min(), x_fit.max(), Nx_sim)
        u_sim = solve_diffusion(tf, D, Cbgr, C0, x_sim, n)

        model = np.interp(x_fit, x_sim, u_sim)
        mask = y_fit > 0
        res = np.zeros_like(y_fit)
        if np.any(mask):
            res[mask] = (model[mask] - y_fit[mask]) / y_fit[mask]
        else:
            res = model - y_fit
        return res
    return residual

# --- GUI Application ---
class DiffusionFitApp:
    def __init__(self, master):
        self.master = master
        prog_name = os.path.basename(__file__).replace("_Final.py", "")
        master.title(f"{prog_name}")
        master.geometry('1024x576')

        # Variables
        self.filename = tk.StringVar(value='')
        self.nx_plot_var = tk.IntVar(value=500)
        self.n_var = tk.DoubleVar(value=DEFAULT_N)
        self.nx_sim_var = tk.IntVar(value=DEFAULT_NX_SIM)
        self.tf = DEFAULT_TF

        self.adv = {
            'C0_init': None,
            'C0_min': None,
            'C0_max': None,
            'Cbgr_init': None,
            'Cbgr_min': None,
            'Cbgr_max': None,
            'D_init': 1e-12,
            'D_min': 1e-16,
            'D_max': 1e-2,
        }

        # Top controls
        topframe = ttk.Frame(master)
        topframe.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(topframe, text='Data file:').grid(row=0, column=0, sticky='w')
        self.entry_file = ttk.Entry(topframe, textvariable=self.filename, width=80)
        self.entry_file.grid(row=0, column=1, padx=4)
        ttk.Button(topframe, text='Browse', command=self.browse_file).grid(row=0, column=2, padx=4)

        # Matrix of spinboxes for Simulation Settings
        matrix_frame = ttk.LabelFrame(topframe, text='Simulation settings')
        matrix_frame.grid(row=2, column=0, columnspan=5, pady=8, sticky='w')

        # Display fitted parameters (to the right of simulation settings)
        param_frame = ttk.LabelFrame(topframe, text='Fitted parameters')
        param_frame.grid(row=2, column=5, padx=10, pady=8, sticky='nw')


        self.param_vars = {
            "D": tk.StringVar(value="--"),
            "C0": tk.StringVar(value="--"),
            "Cbgr": tk.StringVar(value="--"),
        }
        units = {
            "D": "cm2/s",
            "C0": "1/cm3",
            "Cbgr": "1/cm3"
        }

        row = 0
        for name, var in self.param_vars.items():
            # parameter label (e.g. "D:")
            ttk.Label(param_frame, text=name + ":").grid(
                row=row, column=0, sticky="w", padx=(2, 4), pady=2
            )
            # numeric value
            # numeric value as a read-only Entry (copyable)
            entry = tk.Entry(
                param_frame,
                textvariable=var,
                width=30,  # adjust as needed
                justify="center",  # right-align numbers
                fg="black",
                state="readonly"
            )
            entry.grid(row=row, column=1, sticky="e", padx=(2, 1), pady=2)
            # units
            ttk.Label(param_frame, text=units[name]).grid(
                row=row, column=2, sticky="w", padx=(1, 2), pady=2
            )
            row += 1

        # Row 1
        ttk.Label(matrix_frame, text='Annealing time (min):').grid(row=0, column=0, sticky='w')
        self.tf_var = tk.DoubleVar(value=30)
        ttk.Spinbox(matrix_frame, textvariable=self.tf_var, from_=0.01, to=1000000,
                    increment=1, width=8).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(matrix_frame, text='Concentration dependence (n):').grid(row=0, column=2, sticky='w')
        ttk.Spinbox(matrix_frame, textvariable=self.n_var, from_=-10, to=10,
                    increment=0.1, width=8).grid(row=0, column=3, padx=4, pady=2)

        # Row 2
        ttk.Label(matrix_frame, text='Simulation points:').grid(row=1, column=0, sticky='w')
        ttk.Spinbox(matrix_frame, textvariable=self.nx_sim_var, from_=100, to=10000,
                    increment=50, width=8).grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(matrix_frame, text='Plot points:').grid(row=1, column=2, sticky='w')
        ttk.Spinbox(matrix_frame, textvariable=self.nx_plot_var, from_=100, to=20000,
                    increment=100, width=8).grid(row=1, column=3, padx=4, pady=2)

        # Attach Advanced settings and Instructions buttons inside the matrix frame on the right
        ttk.Button(matrix_frame, text='Advanced settings', command=self.open_advanced).grid(row=0, column=4, padx=10,
                                                                                               pady=2, sticky='w')
        ttk.Button(matrix_frame, text='Instructions', command=self.show_instructions).grid(row=1, column=4, padx=10,
                                                                                           pady=2, sticky='w')

        # Buttons
        btnframe = ttk.Frame(master)
        btnframe.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        self.run_button = ttk.Button(btnframe, text='Run fit', command=self.run_fit_thread)
        self.run_button.pack(side=tk.LEFT)

        ttk.Button(btnframe, text='Save last simulation', command=self.save_last_simulation).pack(side=tk.LEFT, padx=6)
        # Checkbox for saving plot after fitting (affects save_last_simulation only)
        self.save_plot_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btnframe, text='Save plot after fit', variable=self.save_plot_var).pack(side=tk.LEFT, padx=6)
        # Author label
        ttk.Label(btnframe, text='Written by Arianna Jaroszynska (Orig. ver. 2023, GUI: 2025)', foreground='gray').pack(side=tk.RIGHT, padx=6)

        # Status bar
        self.status_var = tk.StringVar(value='Ready')
        self.status_label = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Plot area
        plotframe = ttk.Frame(master)
        plotframe.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotframe)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Data containers
        self.x_exp = None
        self.y_exp = None
        self.last_sim_x = None
        self.last_sim_y = None
        self.result = None

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        try:
            self.master.quit()
            self.master.destroy()
        except Exception:
            pass

    def browse_file(self):
        fname = filedialog.askopenfilename(
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt')],
            initialdir=os.getcwd()
        )
        if fname:
            self.filename.set(fname)
            self.load_data()

    def load_data(self):
        fname = self.filename.get()
        if not fname or not os.path.exists(fname):
            messagebox.showerror('Error', 'Please select a valid data file first.')
            return
        try:
            x_exp, y_exp = np.loadtxt(fname, unpack=True)
        except Exception as e:
            messagebox.showerror('Error loading data', str(e))
            return
        x_exp = x_exp * umtocm
        self.x_exp = x_exp
        self.y_exp = y_exp
        self.status_var.set(f'Loaded {os.path.basename(fname)} ({len(x_exp)} points)')
        self.guess_advanced_defaults()
        # draw experimental data immediately
        self.plot_data()

    def guess_advanced_defaults(self):
        if self.x_exp is None:
            return
        y_exp = self.y_exp
        N_edge = max(5, len(y_exp)//10)
        y_start = y_exp[:N_edge]
        y_tail = y_exp[-N_edge:]

        C0_init = np.median(y_start)
        C0_min = np.percentile(y_start, 5)
        C0_max = np.percentile(y_start, 95)

        Cbgr_init = np.median(y_tail)
        Cbgr_min = np.percentile(y_tail, 5)
        Cbgr_max = np.percentile(y_tail, 95)

        self.adv.update({
            'C0_init': C0_init,
            'C0_min': max(C0_min, 1e-30),
            'C0_max': max(C0_max, C0_init*1.1),
            'Cbgr_init': Cbgr_init,
            'Cbgr_min': max(Cbgr_min, 1e-30),
            'Cbgr_max': max(Cbgr_max, Cbgr_init*1.1),
        })

    def open_advanced(self):
        win = tk.Toplevel(self.master)
        win.title('Advanced settings')
        win.grab_set()

        entries = {}
        row = 0
        for key, label in [('C0_init','C0 initial'), ('C0_min','C0 min'), ('C0_max','C0 max')]:
            ttk.Label(win, text=label+':').grid(row=row, column=0, sticky='e', padx=4, pady=2)
            var = tk.StringVar(value=f"{self.adv.get(key, ''):.3e}" if self.adv.get(key) is not None else '')
            entries[key] = var
            ttk.Entry(win, textvariable=var, width=20).grid(row=row, column=1, padx=4, pady=2)
            row += 1

        for key, label in [('Cbgr_init','Cbgr initial'), ('Cbgr_min','Cbgr min'), ('Cbgr_max','Cbgr max')]:
            ttk.Label(win, text=label+':').grid(row=row, column=0, sticky='e', padx=4, pady=2)
            var = tk.StringVar(value=f"{self.adv.get(key, ''):.2E}" if self.adv.get(key) is not None else '')
            entries[key] = var
            ttk.Entry(win, textvariable=var, width=20).grid(row=row, column=1, padx=4, pady=2)
            row += 1

        for key, label in [('D_init','D initial'), ('D_min','D min'), ('D_max','D max')]:
            ttk.Label(win, text=label+':').grid(row=row, column=0, sticky='e', padx=4, pady=2)
            var = tk.StringVar(value=f"{self.adv.get(key, ''):.2E}" if self.adv.get(key) is not None else '')
            entries[key] = var
            ttk.Entry(win, textvariable=var, width=20).grid(row=row, column=1, padx=4, pady=2)
            row += 1

        def save_and_close():
            try:
                for k, v in entries.items():
                    val = float(v.get())
                    self.adv[k] = val
            except Exception as e:
                messagebox.showerror('Invalid value', str(e))
                return
            win.destroy()

        ttk.Button(win, text='Save', command=save_and_close).grid(row=row, column=0, columnspan=2, pady=8)

    def plot_data(self):
        # This function MUST be called from the main thread / Tk event loop
        self.ax.clear()
        if self.x_exp is not None:
            self.ax.semilogy(self.x_exp/umtocm, self.y_exp, 'o', label='Experimental')
        if self.last_sim_x is not None:
            self.ax.plot(self.last_sim_x/umtocm, self.last_sim_y, '-', label='Simulation')
        self.ax.set_xlabel('x [µm]')
        self.ax.set_ylabel('C(x) [cm$^{-3}$]')
        self.ax.grid(True, which='both', ls='--')
        # ✅ Force x-axis to start at 0
        self.ax.set_xlim(left=0)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()


    def save_last_simulation(self):
        # Single robust save function
        if self.last_sim_x is None:
            messagebox.showinfo('No simulation', 'No simulation available to save.')
            return

        base = os.path.splitext(self.filename.get())[0]
        default_name = base + "_fit.txt"

        fname = filedialog.asksaveasfilename(
            initialfile=os.path.basename(default_name),
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt')]
        )
        if not fname:
            return

        # Save simulation data
        header = 'x [um]   C [1/cm3]'
        np.savetxt(fname, np.c_[self.last_sim_x / umtocm, self.last_sim_y], header=header)

        # Save fit results in separate file
        fit_results_file = fname.replace('.txt', '_fit_results.txt')
        with open(fit_results_file, 'w') as f:
            f.write('Fit results:\n')
            f.write(f'Original data file: {os.path.basename(self.filename.get())}\n')
            if self.result is not None:
                D_fitted = 10 ** self.result.params['logD'].value
                C0_fitted = 10 ** self.result.params['logC0'].value
                Cbgr_fitted = 10 ** self.result.params['logCbgr'].value

                # --- Absolute errors ---
                D_error = np.log(10) * D_fitted * (self.result.params['logD'].stderr or 0)
                C0_error = np.log(10) * C0_fitted * (self.result.params['logC0'].stderr or 0)
                Cbgr_error = np.log(10) * Cbgr_fitted * (self.result.params['logCbgr'].stderr or 0)

                # --- Relative uncertainties (percent) ---
                D_uncertainty = 100 * D_error / D_fitted if D_fitted else 0
                C0_uncertainty = 100 * C0_error / C0_fitted if C0_fitted else 0
                Cbgr_uncertainty = 100 * Cbgr_error / Cbgr_fitted if Cbgr_fitted else 0

                f.write(f"D [cm2/s] = {D_fitted:.3E}" + (f" ± {D_error:.3E}" if D_error else "") + f" ({D_uncertainty:.1f}%)\n")
                f.write(f"C0 [1/cm3] = {C0_fitted:.3E}" + (f" ± {C0_error:.3E}" if C0_error else "") + f" ({C0_uncertainty:.1f}%)\n")
                f.write(f"Cbgr [1/cm3] = {Cbgr_fitted:.3E}" + (f" ± {Cbgr_error:.3E}" if Cbgr_error else "") + f" ({Cbgr_uncertainty:.1f}%)\n")


        plot_filename = None
        if self.save_plot_var.get():
            try:
                # Ensure GUI has latest rendering
                self.canvas.draw()
                if hasattr(self.canvas, 'flush_events'):
                    try:
                        self.canvas.flush_events()
                    except Exception:
                        pass
                plot_filename = fname.replace('.txt', '_plot.png')

                # Save current figure size, then temporarily set to 7x5.5 for export only
                original_size = self.fig.get_size_inches()
                self.fig.set_size_inches(7, 5.5)
                self.fig.tight_layout()
                self.fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
                # Restore the original size so Tkinter view is unaffected
                self.fig.set_size_inches(original_size)
                self.fig.tight_layout()
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror('Save plot error', f'Could not save plot: {e}')

        messagebox.showinfo(
            'Saved',
            f"Saved simulation to:\n{os.path.basename(fname)}\nFit results to:\n{os.path.basename(fit_results_file)}"
            + (f"\nPlot to:\n{os.path.basename(plot_filename)}" if plot_filename else "")
        )

    def run_fit_thread(self):
        # Start worker thread that does the heavy numerical work
        t = threading.Thread(target=self._fit_worker, daemon=True)
        t.start()

    def _fit_worker(self):
        # This runs in worker thread. It should NOT call GUI functions directly.
        if self.x_exp is None:
            # schedule error in main thread
            self.master.after(0, lambda: messagebox.showerror('Error', 'No experimental data loaded. Please load a file first.'))
            return
        # disable button in main thread
        self.master.after(0, lambda: self.run_button.config(state=tk.DISABLED))
        self.master.after(0, lambda: self.status_var.set('Starting fit...'))

        try:
            x_exp = self.x_exp
            y_exp = self.y_exp
            Nx_sim = int(self.nx_sim_var.get())
            Nx_plot = int(self.nx_plot_var.get())
            n = float(self.n_var.get())
            # Convert annealing time from minutes to seconds
            tf_seconds = float(self.tf_var.get()) * 60
            if self.adv['C0_init'] is None:
                self.guess_advanced_defaults()

            params = Parameters()
            D_init = float(self.adv.get('D_init', 1e-12))
            D_min = float(self.adv.get('D_min', 1e-16))
            D_max = float(self.adv.get('D_max', 1e-2))
            params.add('logD', value=np.log10(D_init), min=np.log10(D_min), max=np.log10(D_max))

            C0_i = float(self.adv.get('C0_init', max(y_exp[0], 1e-30)))
            C0_min = float(self.adv.get('C0_min', max(C0_i * 0.1, 1e-30)))
            C0_max = float(self.adv.get('C0_max', C0_i * 10))
            params.add('logC0', value=np.log10(C0_i), min=np.log10(max(C0_min, 1e-40)),
                       max=np.log10(max(C0_max, C0_i * 1.001)))

            Cbgr_i = float(self.adv.get('Cbgr_init', max(y_exp[-1], 1e-30)))
            Cbgr_min = float(self.adv.get('Cbgr_min', max(Cbgr_i * 0.1, 1e-30)))
            Cbgr_max = float(self.adv.get('Cbgr_max', Cbgr_i * 10))
            params.add('logCbgr', value=np.log10(Cbgr_i), min=np.log10(max(Cbgr_min, 1e-40)),
                       max=np.log10(max(Cbgr_max, Cbgr_i * 1.001)))

            # update status
            self.master.after(0, lambda: self.status_var.set('Running fit (this may take a while)...'))

            residual = make_residual(x_exp, y_exp, Nx_sim, tf_seconds, n)
            result = minimize(residual, params)

            # Fitted values
            D_fitted = 10 ** result.params['logD'].value
            C0_fitted = 10 ** result.params['logC0'].value
            Cbgr_fitted = 10 ** result.params['logCbgr'].value

            # Uncertainties (absolute values)
            D_error = np.log(10) * D_fitted * (result.params['logD'].stderr or 0)
            C0_error = np.log(10) * C0_fitted * (result.params['logC0'].stderr or 0)
            Cbgr_error = np.log(10) * Cbgr_fitted * (result.params['logCbgr'].stderr or 0)

            # Relative uncertainties (%)
            D_uncertainty = 100 * D_error / D_fitted if D_fitted else 0
            C0_uncertainty = 100 * C0_error / C0_fitted if C0_fitted else 0
            Cbgr_uncertainty = 100 * Cbgr_error / Cbgr_fitted if Cbgr_fitted else 0

            # Prepare high-resolution simulation for plotting (done in worker)
            Nx_plot = max(50, Nx_sim)
            x_sim_plot = np.linspace(x_exp.min(), x_exp.max(), Nx_plot)
            u_sim_plot = solve_diffusion(tf_seconds, D_fitted, Cbgr_fitted, C0_fitted, x_sim_plot, n)

            # Schedule GUI update / finalize on main thread
            def finalize():
                # store results
                self.result = result
                self.param_vars["D"].set(f"{D_fitted:.3E}" + (f" ± {D_error:.3E} " if D_error else "") + (f"({D_uncertainty:.1f}%)" if Cbgr_error else ""))
                self.param_vars["C0"].set(f"{C0_fitted:.3E}" + (f" ± {C0_error:.3E} " if C0_error else "") + (f"({C0_uncertainty:.1f}%)" if Cbgr_error else ""))
                self.param_vars["Cbgr"].set(f"{Cbgr_fitted:.3E}" + (f" ± {Cbgr_error:.3E} " if Cbgr_error else "") + (f"({Cbgr_uncertainty:.1f}%)" if Cbgr_error else ""))

                self.last_sim_x = x_sim_plot
                self.last_sim_y = u_sim_plot

                # update plot and status
                self.plot_data()
                self.status_var.set('Fit completed successfully.')

                # popup with parameters
                try:
                    messagebox.showinfo(
                        'Fit results',
                        f"Fitted parameters:\n"
                        f"D = {D_fitted:.3E} ± {D_error:.3E} cm2/s ({D_uncertainty:.1f}%)\n"
                        f"C0 = {C0_fitted:.3E} ± {C0_error:.3E} 1/cm3 ({C0_uncertainty:.1f}%)\n"
                        f"Cbgr = {Cbgr_fitted:.3E} ± {Cbgr_error:.3E} 1/cm3 ({Cbgr_uncertainty:.1f}%)\n"
                    )
                except Exception:
                    pass

                # re-enable run button
                self.run_button.config(state=tk.NORMAL)

            self.master.after(0, finalize)

        except Exception as e:
            # schedule error display and re-enable button
            def show_err():
                messagebox.showerror('Fit error', str(e))
                self.run_button.config(state=tk.NORMAL)
                self.status_var.set('Ready')
            self.master.after(0, show_err)

    def show_instructions(self):
        win = tk.Toplevel(self.master)
        win.title('Instructions')
        win.geometry('640x480')
        # Create a frame to hold text and scrollbar
        frame = tk.Frame(win)
        frame.pack(expand=True, fill='both')
        # Scrollbar
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        # Text widget
        text = tk.Text(frame, wrap='word', yscrollcommand=scrollbar.set)
        text.pack(expand=True, fill='both')
        # Link scrollbar to text widget
        scrollbar.config(command=text.yview)
        doc = (
        "GENERAL OVERVIEW\n\n"
        "This program fits diffusion profiles based on a numerical solution to Fick’s second law using finite difference method. "
        "It supports concentration-dependent diffusion (parameter n) and automatically optimizes "
        "the key physical parameters D, C0, and Cbgr using non-linear least squares.\n\n"

        "DATA FORMAT\n\n"
        "Input files must be plain text (.txt) with two tab-separated columns:\n\n"
        "1st column → depth x [µm]\n"
        "2nd column → concentration C [1/cm3]\n\n"
        "Example:\n"
        "0.5\t1e18\n"
        "0.75\t8e17\n"
        "1.0\t6e17\n\n"
        "The second column does not have to be in 1e17 format, it can also be 5000 etcetera.\n\n"
            
        "SIMULATION SETTINGS\n\n"
        
        "• Annealing time (min) → Duration of the diffusion process.\n\n"
        
        "• Concentration dependence (n) → Concentration dependence exponent. \n"
        "Generally, n values are integers (negative too) like -2, -1, 0, 1, 2, 3). "
        "They are correlated with governing diffusion mechanisms. " 
        "To find the best fit for your data, it is best to consult the scientific literature. "
        "If the literature is lacking, try multiple fits with various n to find the best fit.\n\n" 
        "Example values:\n"
        "n = 0 gives standard erfc profile diffusion (default).\n"
        "n = 1 for profiles like Mg in GaN diffusion\n"
        "n = 3 for steep, cliff-like profiles\n\n"
            
        "• Simulation points → Number of grid points for simulation. Larger = higher accuracy, slower.\n\n" 
        "Recommended values:\n"
        ">200 if you do not mind waiting for a long time for results\n"
        "200 for still fast and good results\n"
        "100 for decent results. \n\n"
            
        "• Plot points → Number of points for final plot dataset.\n\n" 
        "Recommended values:\n"
        ">1000 for if you really need a lot of data\n"
        "1000 for a pretty curve\n"
        "500 for decent looking curve (default)\n"
        "100 for quick testing\n\n"

        "ADVANCED SETTINGS WINDOW\n\n"
        "The 'Advanced Settings' button opens a separate dialog where you can fine-tune "
        "the parameter boundaries and initial guesses for the fitting process. These help "
        "stabilize or constrain the optimizer if your data are unusual or noisy.\n\n"
        "Parameters available:\n"
        "• C0 initial / min / max  → Surface concentration C0 at the left boundary.\n"
        "   - 'initial': the optimizer's starting guess.\n"
        "   - 'min' and 'max': define the allowed range of variation.\n\n"
        "• Cbgr initial / min / max  → Background concentration (deep in the material).\n"
        "   - Typically the lowest part of your profile.\n\n"
        "• D initial / min / max  → Diffusion coefficient bounds.\n"
        "   - Controls the diffusion speed.\n"
        "   - Reasonable starting value is 1e-7 cm²/s for many dopants.\n\n"

        "SAVING RESULTS\n\n"
        "- After fitting, press 'Save last simulation' to export:\n"
        "   • *_fit.txt → simulated profile (x, C)\n"
        "   • *_fit_results.txt → fitted parameter summary\n"
        "   • *_plot.png → high-resolution plot (if 'Save plot after fit' is checked)\n\n"

        "OUTPUT PLOT\n\n"
        "- The plot displays experimental data (points) and fitted simulation (line).\n"
        "- The x-axis is in µm, y-axis in 1/cm3 (log scale).\n"
        "- You can save the plot automatically by ticking the checkbox.\n\n"

        "AUTHOR\n\n"
        "Hello! \n\n"
        "I wrote this back in 2023 (original version).\n"
        "Added GUI in 2025.\n\n"
        "The source code is located on my github repository: \n"
        "https://github.com/Arianna-Projects/DiffusionFitting\n\n"
        "Thank you for using my program :)\n\n"
        "Have a great day!\n\n"
        "Love, Arianna"
        )
        text.insert('1.0', doc)
        text.config(state='disabled')


if __name__ == '__main__':
    root = tk.Tk()
    app = DiffusionFitApp(root)
    root.mainloop()


