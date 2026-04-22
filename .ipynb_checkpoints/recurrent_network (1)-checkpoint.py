# -*- coding: utf-8 -*-
#
# brunel_delta_nest.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Random balanced network (delta synapses)
----------------------------------------

This script simulates an excitatory and an inhibitory population on
the basis of the network used in [1]_

When connecting the network, customary synapse models are used, which
allow for querying the number of created synapses. Using spike
recorders, the average firing rates of the neurons in the populations
are established. The building as well as the simulation time of the
network are recorded.

References
~~~~~~~~~~

.. [1] Brunel N (2000). Dynamics of sparsely connected networks of excitatory and
       inhibitory spiking neurons. Journal of Computational Neuroscience 8,
       183-208.

"""

###############################################################################
# Import all necessary modules for simulation, analysis and plotting.

import time
import nest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def positive_normal(mu, std, size, resolution):
    """This function draws random delays drawn from a Gaussian Distribution
   ensuring that the delays are bigger than the temporal resolution of the simulations.
   mu: mean value 
   std: standard deviation
   size: number of delays (i.e. #connections)"""
    
    delays = np.random.normal(mu, std, size)
    
    while np.any(delays < resolution):
        delays[delays < resolution] = np.random.normal(mu, std, np.sum(delays < resolution))
    return delays

def sim_brunel(dt=0.25,
               simtime=1000.0,
               delay=0.25,
               g=4.0,
               eta=2.0,
               epsilon=0.1,
               NE=800,
               NI=200,
               J=0.1,
               num_vp=8,
               seed=1,
               print_report=True,
               a=1000,
               f1=1000,
               std=250,
               beat=20,
               noisy=True,
               second_sine=True,
               neuron_type="iaf_psc_delta",
               measure_from_A=False,
               theta=-50.0,
               tau_syn_ex=2):
    
    """--------------Arguments---------------------------------
        epsilon : connection probability
        order : number of inhibitory synapses. excitatory = 4*order
        J = postsynaptic amplitude in mV '
        g: ratio inhibitory weight/excitatory weight
        eta: external rate relative to threshold rate
        
        --------------Returns--------------------------------
        dataframe with results 
        """ 

    nest.ResetKernel()

    
    ###############################################################################
    # Assigning the current time to a variable in order to determine the build
    # time of the network.
    
    startbuild = time.time()
    
    ###############################################################################
    # Assigning the simulation parameters to variables.
    
    dt = 0.25  # the resolution in ms
    delay = 1.5  # synaptic delay in ms
    f2 = f1 + beat
    
    ###############################################################################
   
    ###############################################################################
    # Definition of the number of neurons in the network and the number of neurons
    # recorded from
    
    N_neurons = NE + NI + 1 # number of neurons in total
    
    ###############################################################################
    # Definition of connectivity parameters
    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron
    
    ###############################################################################
    # Initialization of the parameters of the integrate and fire neuron and the
    # synapses. The parameters of the neuron are stored in a dictionary.
    
    tauMem = 10.0  # time constant of membrane potential in ms
    neuron_params = {"C_m": 100.0,
                     "tau_m": tauMem,
                     "t_ref": 2.0,
                     "E_L": -60.0,
                     "V_reset": -60.0,
                     "V_m": -60.0,
                     "V_th": theta,
                     "tau_syn_ex": tau_syn_ex}
    
    if neuron_type == "iaf_cond_exp":
        neuron_params.pop("tau_m")
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential
    
    ###############################################################################
    # Definition of threshold rate, which is the external rate needed to fix the
    # membrane potential around its threshold, the external firing rate and the
    # rate of the poisson generator which is multiplied by the in-degree CE and
    # converted to Hz by multiplication by 1000.
    
    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    
    ###############################################################################
    # Configuration of the simulation kernel by the previously defined time
    # resolution used in the simulation. Setting ``print_time`` to `True` prints the
    # already processed simulation time as well as its percentage of the total
    # simulation time.

    nest.set_verbosity("M_WARNING")
    nest.resolution = dt
    nest.total_num_virtual_procs = num_vp
    nest.print_time = False
    nest.rng_seed = seed
    #nest.total_num_virtual_procs = num_vp
    nest.SetKernelStatus({
    "min_delay": 0.3,                 
    "max_delay": 150.0 })
    print("Building network")
    
    ###############################################################################
    # Creation of the nodes using ``Create``. 
    
    nodes_ex = nest.Create(neuron_type, NE+1, params=neuron_params)
    if NI != 0:
        nodes_in = nest.Create(neuron_type, NI, params=neuron_params)
        
    multimeter = nest.Create("multimeter", 1,
            params={
                "interval": dt,
                "record_from": ["V_m"],
    
            })
    
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")
    spikes = nest.Create("spike_recorder")
    
    # sinusoidal input current
    sine1 = nest.Create("ac_generator", 
                        params={"amplitude": a, "frequency": f1})
    if second_sine:
        sine2 = nest.Create("ac_generator", 
                            params={"amplitude": a, "frequency": f2})
    
    ###############################################################################
    # Configuration of the spike recorders recording excitatory and inhibitory
    # spikes by sending parameter dictionaries to ``set``. Setting the property
    # `record_to` to *"ascii"* ensures that the spikes will be recorded to a file,
    # whose name starts with the string assigned to the property `label`.
    
    #espikes.set(label="brunel-py-ex", record_to="ascii")
    #ispikes.set(label="brunel-py-in", record_to="ascii")
    
    print("Connecting devices")
    
    ###############################################################################
    # Definition of a synapse using ``CopyModel``, which expects the model name of
    # a pre-defined synapse, the name of the customary synapse and an optional
    # parameter dictionary. The parameters defined in the dictionary will be the
    # default parameter for the customary synapse. Here we define one synapse for
    # the excitatory and one for the inhibitory connections giving the
    # previously defined weights and equal delays.
    
    nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex, "delay" : 0.5
                })

    nest.CopyModel("static_synapse", "inhibitory",
               {"weight": J_in, "delay" : 0.5})
    
    
    ###############################################################################
    # Connecting the previously defined poisson generator to the excitatory and
    # inhibitory neurons using the excitatory synapse. Since the poisson
    # generator is connected to all neurons in the population the default rule
    # (# ``all_to_all``) of ``Connect`` is used. The synaptic properties are inserted
    # via ``syn_spec`` which expects a dictionary when defining multiple variables
    # or a string when simply using a pre-defined synapse.
 
    if noisy:
        noise_generator = nest.Create("noise_generator",params={"mean": 0.0, "std": 250, "dt" : 0.5})
        for i, n in enumerate(nodes_ex):
            nest.Connect(noise_generator, n, syn_spec={'delay': np.random.normal(loc=100.0, scale=4.0, size=None)})
   # 
   # nest.Connect(sine1, nodes_ex + nodes_in)
    
   #if noisy:
   #    noise_generator = nest.Create("noise_generator", params={"mean": 0.0, "std": std, "dt": 0.5})
   #    # normaldist µ=100 ms, σ=4 ms
   #    delay_dist = nest.random.normal(100.0, 4.0)     
   #    syn_spec = {"delay": delay_dist}
   #    if NI != 0:
   #        nest.Connect(noise_generator, nodes_ex + nodes_in, syn_spec=syn_spec)
   #    else:
   #        nest.Connect(noise_generator, nodes_ex, syn_spec=syn_spec)
        
      #  conns  = nest.GetConnections(noise_generator, nodes_ex)
      #  delays = conns.get('delay')        # numpy-array
      #  print(delays[:10])   
    
    if NI != 0:
        nest.Connect(sine1, nodes_ex[:-1] + nodes_in)
        if second_sine:
            nest.Connect(sine2, nodes_ex[:-1] + nodes_in)
    else:
        nest.Connect(sine1, nodes_ex[:-1])
        if second_sine:
            nest.Connect(sine2, nodes_ex[:-1])
    
    

    
    
    ###############################################################################
    # Connecting the first ``N_rec`` nodes of the excitatory and inhibitory
    # population to the associated spike recorders using excitatory synapses.
    # Here the same shortcut for the specification of the synapse as defined
    # above is used.
    
    nest.Connect(nodes_ex[-1], espikes, syn_spec="excitatory")
    
    if measure_from_A:
        nest.Connect(multimeter, nodes_ex[0], "one_to_one")
    else:
        nest.Connect(multimeter, nodes_ex[-1], "one_to_one")
    
    if NI != 0:
        nest.Connect(nodes_in, ispikes, syn_spec="excitatory")
      
    
    if measure_from_A:
         nest.Connect(nodes_ex[0], spikes, syn_spec="excitatory")
        
    else:
        nest.Connect(nodes_ex[-1], spikes, syn_spec="excitatory")
    
    print("Connecting network")
    
    print("Excitatory connections")
    
    ###############################################################################
    # Connecting the excitatory population to all neurons using the pre-defined
    # excitatory synapse. Beforehand, the connection parameter are defined in a
    # dictionary. Here we use the connection rule ``fixed_indegree``,
    # which requires the definition of the indegree. Since the synapse
    # specification is reduced to assigning the pre-defined excitatory synapse it
    # suffices to insert a string.
    
    
    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}

    
    if NI == 0:
         nest.Connect(nodes_ex, nodes_ex, conn_params_ex, "excitatory")
    else:
        nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")
    
    conns_ex = nest.GetConnections(source=nodes_ex)
    n = len(conns_ex)
    delays = np.clip(np.random.normal(loc=1.5, scale=0.25, size=n), 0.25, None)
#
    conns_ex.set({"delay": delays})
    
    print("Inhibitory connections")
    
    ###############################################################################
    # Connecting the inhibitory population to all neurons using the pre-defined
    # inhibitory synapse. The connection parameters as well as the synapse
    # parameters are defined analogously to the connection from the excitatory
    # population defined above.

    """ DELAYS HER? """
    if NI != 0:
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
        nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

    
    ###############################################################################
    # Storage of the time point after the buildup of the network in a variable.
    
    endbuild = time.time()
    
    ###############################################################################
    # Simulation of the network.
    
    print("Simulating")
    
    nest.Simulate(simtime)
    
    ###############################################################################
    # Storage of the time point after the simulation of the network in a variable.
    
    endsimulate = time.time()
    
    ###############################################################################
    # Reading out the total number of spikes received from the spike recorder
    # connected to the excitatory population and the inhibitory population.
  
    events_ex = espikes.n_events
    
    rate_in = 0
    events_in = 0
    if NI !=0:
        events_in = ispikes.n_events
        rate_in = events_in / simtime * 1000 
    
    print("Number of spikes", events_ex + events_in)
    
    ###############################################################################
    # Calculation of the average firing rate of the excitatory and the inhibitory
    # neurons by dividing the total number of recorded spikes by the number of
    # neurons recorded from and the simulation time. The multiplication by 1000.0
    # converts the unit 1/ms to 1/s=Hz.
    
    rate_ex = events_ex / simtime * 1000 
    
    
    ###############################################################################
    # Reading out the number of connections established using the excitatory and
    # inhibitory synapse model. The numbers are summed up resulting in the total
    # number of synapses.
    
    num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                    nest.GetDefaults("inhibitory")["num_connections"])
    
    ###############################################################################
    # Establishing the time it took to build and simulate the network by taking
    # the difference of the pre-defined time variables.
    
    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild
    
    ###############################################################################
    # Printing the network properties, firing rates and building times.
    
    print("Brunel network simulation (Python)")
    print(f"Number of neurons : {N_neurons}")
    print(f"Number of synapses: {num_synapses}")
    print(f"       Exitatory  : {int(CE * N_neurons) + N_neurons}")
    print(f"       Inhibitory : {int(CI * N_neurons)}")
    print(f"Excitatory rate   : {rate_ex:.2f} Hz")
    print(f"Inhibitory rate   : {rate_in:.2f} Hz")
    print(f"Building time     : {build_time:.2f} s")
    print(f"Simulation time   : {sim_time:.2f} s")
    
    ###############################################################################
    # Plot a raster of the excitatory neurons and a histogram.
    
    #nest.raster_plot.from_device(espikes, hist=True)
    #plt.show()
    exc_spikes = espikes.get('events')
    inh_spikes = ispikes.get('events')
    spikes_neuron1 = spikes.get('events')
    
    exc_ts = np.array(exc_spikes["times"])
    inh_ts = np.array(inh_spikes["times"])
    spike_times = np.array(spikes_neuron1["times"])
    
    
    mm_data = multimeter.get("events")
        
    V_m = mm_data["V_m"]
    times = mm_data["times"]
    
    
    results = {
        "exc_spike_times": exc_ts,
        "inh_spike_times" : inh_ts,
        "spike_times" : spike_times,
        "Vm": V_m,
        "SD for Vm" : np.std(V_m),
        "times" : times,
        "Number_of_spikes" : len(exc_ts)+len(inh_ts),
        "Spike_rate" : (len(exc_ts)+len(inh_ts))/sim_time*1000 #spikes/sec

    }

    
    return results
    