import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import Configuration, config_error
from environment import build_hospital
from infection import find_nearby, infect, recover_or_die, compute_mortality,\
healthcare_infection_correction
from motion import update_positions, out_of_bounds, update_randoms,\
get_motion_parameters
from path_planning import go_to_location, set_destination, check_at_destination,\
keep_at_destination, reset_destinations
from population import initialize_population, initialize_destination_matrix,\
set_destination_bounds, save_data, save_population, Population_trackers
from visualiser import build_fig, draw_tstep, set_style, plot_sir

import tkinter as tk
from tkinter import ttk
from threading import Thread
import csv

class Simulation():
    def __init__(self, *args, **kwargs):
        #load default config data
        self.config = config
        self.frame = 0

        #initialize default population
        self.population_init()

        self.pop_tracker = Population_trackers()

        #initalise destinations vector
        self.destinations = initialize_destination_matrix(self.config.pop_size, 1)


    def reinitialise(self):
        '''reset the simulation'''

        self.frame = 0
        self.population_init()
        self.pop_tracker = Population_trackers()
        self.destinations = initialize_destination_matrix(self.config.pop_size, 1)


    def population_init(self):
        '''(re-)initializes population'''
        self.population = initialize_population(self.config, self.config.mean_age,
                                                self.config.max_age, self.config.xbounds,
                                                self.config.ybounds)


    def tstep(self):
        '''
        takes a time step in the simulation
        '''

        if self.frame == 0 and self.config.visualise:
            #initialize figure
            self.fig, self.spec, self.ax1, self.ax2 = build_fig(self.config)

        #check destinations if active
        #define motion vectors if destinations active and not everybody is at destination
        active_dests = len(self.population[self.population[:,11] != 0]) # look op this only once

        if active_dests > 0 and len(self.population[self.population[:,12] == 0]) > 0:
            self.population = set_destination(self.population, self.destinations)
            self.population = check_at_destination(self.population, self.destinations,
                                                   wander_factor = self.config.wander_factor_dest,
                                                   speed = self.config.speed)

        if active_dests > 0 and len(self.population[self.population[:,12] == 1]) > 0:
            #keep them at destination
            self.population = keep_at_destination(self.population, self.destinations,
                                                  self.config.wander_factor)

        #out of bounds
        #define bounds arrays, excluding those who are marked as having a custom destination
        if len(self.population[:,11] == 0) > 0:
            _xbounds = np.array([[self.config.xbounds[0] + 0.02, self.config.xbounds[1] - 0.02]] * len(self.population[self.population[:,11] == 0]))
            _ybounds = np.array([[self.config.ybounds[0] + 0.02, self.config.ybounds[1] - 0.02]] * len(self.population[self.population[:,11] == 0]))
            self.population[self.population[:,11] == 0] = out_of_bounds(self.population[self.population[:,11] == 0],
                                                                        _xbounds, _ybounds)

        #set randoms
        if self.config.lockdown:
            if len(self.pop_tracker.infectious) == 0:
                mx = 0
            else:
                mx = np.max(self.pop_tracker.infectious)

            if len(self.population[self.population[:,6] == 1]) >= len(self.population) * self.config.lockdown_percentage or\
               mx >= (len(self.population) * self.config.lockdown_percentage):
                #reduce speed of all members of society
                self.population[:,5] = np.clip(self.population[:,5], a_min = None, a_max = 0.001)
                #set speeds of complying people to 0
                self.population[:,5][self.config.lockdown_vector == 0] = 0
            else:
                #update randoms
                self.population = update_randoms(self.population, self.config.pop_size, self.config.speed)
        else:
            #update randoms
            self.population = update_randoms(self.population, self.config.pop_size, self.config.speed)

        #for dead ones: set speed and heading to 0
        self.population[:,3:5][self.population[:,6] == 3] = 0

        #update positions
        self.population = update_positions(self.population)

        #find new infections
        self.population, self.destinations = infect(self.population, self.config, self.frame,
                                                    send_to_location = self.config.self_isolate,
                                                    location_bounds = self.config.isolation_bounds,
                                                    destinations = self.destinations,
                                                    location_no = 1,
                                                    location_odds = self.config.self_isolate_proportion)

        #recover and die
        self.population = recover_or_die(self.population, self.frame, self.config)

        #send cured back to population if self isolation active
        #perhaps put in recover or die class
        #send cured back to population
        self.population[:,11][self.population[:,6] == 2] = 0

        #update population statistics
        self.pop_tracker.update_counts(self.population)

        #visualise
        if self.config.visualise:
            draw_tstep(self.config, self.population, self.pop_tracker, self.frame,
                       self.fig, self.spec, self.ax1, self.ax2)

        #report stuff to console
        sys.stdout.write('\r')
        sys.stdout.write('%i: healthy: %i, infected: %i, immune: %i, in treatment: %i, \
dead: %i, of total: %i' %(self.frame, self.pop_tracker.susceptible[-1], self.pop_tracker.infectious[-1],
                        self.pop_tracker.recovered[-1], len(self.population[self.population[:,10] == 1]),
                        self.pop_tracker.fatalities[-1], self.config.pop_size))
        
        healthy = self.pop_tracker.susceptible[-1]
        infected = self.pop_tracker.infectious[-1]
        immune = self.pop_tracker.recovered[-1]
        in_treatment = len(self.population[self.population[:, 10] == 1])  # Assuming column 10 tracks treatment
        dead = self.pop_tracker.fatalities[-1]
        total = self.config.pop_size
        
        # Define file name for saving CSV
        file_name = f"simulation_report.csv"

        # Save data to CSV
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header only for the first frame
            if self.frame == 0:
                header = ["Frame", "Healthy", "Infected", "Immune", "In Treatment", "Dead", "Total"]
                writer.writerow(header)

            # Write the current frame's data
            row = [self.frame, healthy, infected, immune, in_treatment, dead, total]
            writer.writerow(row)

        print(f"Report saved for frame {self.frame} to {file_name}")

        #save popdata if required
        if self.config.save_pop and (self.frame % self.config.save_pop_freq) == 0:
            save_population(self.population, self.frame, self.config.save_pop_folder)
        #run callback
        self.callback()

        #update frame
        self.frame += 1


    def callback(self):
        '''placeholder function that can be overwritten.

        By ovewriting this method any custom behaviour can be implemented.
        The method is called after every simulation timestep.
        '''

        if self.frame == 50:
            print('\ninfecting patient zero')
            self.population[0][6] = 1
            self.population[0][8] = 50
            self.population[0][10] = 1


    def run(self):
        '''run simulation'''

        print(self.config.pop_size)
        
        i = 0

        while i < self.config.simulation_steps:
            try:
                self.tstep()
            except KeyboardInterrupt:
                print('\nCTRL-C caught, exiting')
                sys.exit(1)

            #check whether to end if no infecious persons remain.
            #check if self.frame is above some threshold to prevent early breaking when simulation
            #starts initially with no infections.
            if self.config.endif_no_infections and self.frame >= 500:
                if len(self.population[(self.population[:,6] == 1) |
                                       (self.population[:,6] == 4)]) == 0:
                    i = self.config.simulation_steps

        if self.config.save_data:
            save_data(self.population, self.pop_tracker)

        #report outcomes
        print('\n-----stopping-----\n')
        print('total timesteps taken: %i' %self.frame)
        print('total dead: %i' %len(self.population[self.population[:,6] == 3]))
        print('total recovered: %i' %len(self.population[self.population[:,6] == 2]))
        print('total infected: %i' %len(self.population[self.population[:,6] == 1]))
        print('total infectious: %i' %len(self.population[(self.population[:,6] == 1) |
                                                          (self.population[:,6] == 4)]))
        print('total unaffected: %i' %len(self.population[self.population[:,6] == 0]))


    def plot_sir(self, size=(6,3), include_fatalities=False,
                 title='S-I-R plot of simulation'):
        plot_sir(self.config, self.pop_tracker, size, include_fatalities,
                 title)


class SimulationApp:
    def __init__(self, config):
        self.config = config  # Pass Configuration instance to the GUI
        self.root = tk.Tk()
        self.root.title("Simulation Configuration")
        self.create_widgets()
        
    def create_widgets(self):
        """
        Create Tkinter widgets and bind them to configuration parameters.
        """
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="EW")

        # Population Size (Slider + Input Field)
        ttk.Label(frame, text="Population Size").grid(row=0, column=0, sticky="W")
        self.pop_size_var = tk.IntVar(value=self.config.pop_size)
        pop_size_slider = tk.Scale(frame, from_=100, to=10000, variable=self.pop_size_var, orient="horizontal")
        pop_size_slider.grid(row=0, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.pop_size_var).grid(row=0, column=2, sticky="W")
        
        pop_size_entry = ttk.Entry(frame, textvariable=self.pop_size_var)
        pop_size_entry.grid(row=0, column=3, sticky="EW")
        
        # Infection Range (Slider + Input Field)
        ttk.Label(frame, text="Infection Range").grid(row=1, column=0, sticky="W")
        self.infection_range_var = tk.DoubleVar(value=self.config.infection_range)
        infection_range_slider = tk.Scale(frame, from_=0.001, to=0.1, resolution=0.001, variable=self.infection_range_var, orient="horizontal")
        infection_range_slider.grid(row=1, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.infection_range_var).grid(row=1, column=2, sticky="W")

        infection_range_entry = ttk.Entry(frame, textvariable=self.infection_range_var)
        infection_range_entry.grid(row=1, column=3, sticky="EW")

        # Speed (Slider + Input Field)
        ttk.Label(frame, text="Speed").grid(row=2, column=0, sticky="W")
        self.speed_var = tk.DoubleVar(value=self.config.speed)
        speed_slider = tk.Scale(frame, from_=0.01, to=1.0, resolution=0.01, variable=self.speed_var, orient="horizontal")
        speed_slider.grid(row=2, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.speed_var).grid(row=2, column=2, sticky="W")

        speed_entry = ttk.Entry(frame, textvariable=self.speed_var)
        speed_entry.grid(row=2, column=3, sticky="EW")

        # Infection Chance (Slider + Input Field)
        ttk.Label(frame, text="Infection Chance").grid(row=3, column=0, sticky="W")
        self.infection_chance_var = tk.DoubleVar(value=self.config.infection_chance)
        infection_chance_slider = tk.Scale(frame, from_=0.01, to=1.0, resolution=0.01, variable=self.infection_chance_var, orient="horizontal")
        infection_chance_slider.grid(row=3, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.infection_chance_var).grid(row=3, column=2, sticky="W")

        infection_chance_entry = ttk.Entry(frame, textvariable=self.infection_chance_var)
        infection_chance_entry.grid(row=3, column=3, sticky="EW")

        # Mortality Chance (Slider + Input Field)
        ttk.Label(frame, text="Mortality Chance").grid(row=5, column=0, sticky="W")
        self.mortality_chance_var = tk.DoubleVar(value=self.config.mortality_chance)
        mortality_chance_slider = tk.Scale(frame, from_=0.01, to=1.0, resolution=0.01, variable=self.mortality_chance_var, orient="horizontal")
        mortality_chance_slider.grid(row=5, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.mortality_chance_var).grid(row=5, column=2, sticky="W")

        mortality_chance_entry = ttk.Entry(frame, textvariable=self.mortality_chance_var)
        mortality_chance_entry.grid(row=5, column=3, sticky="EW")

        # Healthcare Capacity (Slider + Input Field)
        ttk.Label(frame, text="Healthcare Capacity").grid(row=6, column=0, sticky="W")
        self.healthcare_capacity_var = tk.IntVar(value=self.config.healthcare_capacity)
        healthcare_capacity_slider = tk.Scale(frame, from_=10, to=1000, variable=self.healthcare_capacity_var, orient="horizontal")
        healthcare_capacity_slider.grid(row=6, column=1, sticky="EW")
        # ttk.Label(frame, textvariable=self.healthcare_capacity_var).grid(row=6, column=2, sticky="W")

        healthcare_capacity_entry = ttk.Entry(frame, textvariable=self.healthcare_capacity_var)
        healthcare_capacity_entry.grid(row=6, column=3, sticky="EW")

        # # Treatment Factor (Slider + Input Field)
        # ttk.Label(frame, text="Treatment Factor").grid(row=7, column=0, sticky="W")
        # self.treatment_factor_var = tk.DoubleVar(value=self.config.treatment_factor)
        # treatment_factor_slider = tk.Scale(frame, from_=0.1, to=10.0, resolution=0.1, variable=self.treatment_factor_var, orient="horizontal")
        # treatment_factor_slider.grid(row=7, column=1, sticky="EW")
        # # ttk.Label(frame, textvariable=self.treatment_factor_var).grid(row=7, column=2, sticky="W")

        # treatment_factor_entry = ttk.Entry(frame, textvariable=self.treatment_factor_var)
        # treatment_factor_entry.grid(row=7, column=3, sticky="EW")

        # # No Treatment Factor (Slider + Input Field)
        # ttk.Label(frame, text="No Treatment Factor").grid(row=8, column=0, sticky="W")
        # self.no_treatment_factor_var = tk.DoubleVar(value=self.config.no_treatment_factor)
        # no_treatment_factor_slider = tk.Scale(frame, from_=0.1, to=10.0, resolution=0.1, variable=self.no_treatment_factor_var, orient="horizontal")
        # no_treatment_factor_slider.grid(row=8, column=1, sticky="EW")
        # # ttk.Label(frame, textvariable=self.no_treatment_factor_var).grid(row=8, column=2, sticky="W")

        # no_treatment_factor_entry = ttk.Entry(frame, textvariable=self.no_treatment_factor_var)
        # no_treatment_factor_entry.grid(row=8, column=3, sticky="EW")

        # Automatically update configuration when sliders change
        self.pop_size_var.trace("w", lambda *args: self.update_config())
        self.infection_range_var.trace("w", lambda *args: self.update_config())
        self.speed_var.trace("w", lambda *args: self.update_config())
        self.infection_chance_var.trace("w", lambda *args: self.update_config())
        self.mortality_chance_var.trace("w", lambda *args: self.update_config())
        self.healthcare_capacity_var.trace("w", lambda *args: self.update_config())
        # self.treatment_factor_var.trace("w", lambda *args: self.update_config())
        # self.no_treatment_factor_var.trace("w", lambda *args: self.update_config())

        # Run Simulation Button
        run_button = ttk.Button(frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=10, column=0, columnspan=4, pady=10)


    def update_config(self):
        """
        Update the configuration instance based on the GUI inputs.
        This will be automatically triggered when a slider is moved.
        """
        try:
            # Update configuration values directly from the sliders
            self.config.pop_size = self.pop_size_var.get()
            self.config.infection_range = self.infection_range_var.get()
            self.config.speed = self.speed_var.get()
            self.config.infection_chance = self.infection_chance_var.get()
            self.config.mortality_chance = self.mortality_chance_var.get()
            self.config.healthcare_capacity = self.healthcare_capacity_var.get()
            # self.config.treatment_factor = self.treatment_factor_var.get()
            # self.config.no_treatment_factor = self.no_treatment_factor_var.get()

            print("Configuration updated successfully:")
            for attr, value in vars(self.config).items():
                print(f"  {attr}: {value}")
        except Exception as e:
            print(f"Error updating configuration: {e}")

    def run_simulation(self):
        """
        Run the simulation with the current configuration.
        """
        self.update_config()  # Ensure the configuration is up-to-date
        
        # Use root.after() to safely destroy the Tkinter window in the main thread
        self.root.after(0, self.start_simulation_thread)

    def start_simulation_thread(self):
        """Start the simulation in a separate thread."""
        # Close the Tkinter window (this should be done in the main thread)
        self.root.destroy()

        # Now run the simulation in a separate thread
        def simulation_thread():
            simulation = Simulation(self.config)  # Use the updated config object
            simulation.run()

        thread = Thread(target=simulation_thread)
        thread.start()

    def run(self):
        """
        Start the Tkinter main loop.
        """
        self.root.mainloop()

# Example Usage
if __name__ == "__main__":
    # Create a configuration object with default values
    config = Configuration(
        simulation_steps=10000,
        endif_no_infections=True,
    )

    # Pass it to the Tkinter app
    app = SimulationApp(config)
    app.run()