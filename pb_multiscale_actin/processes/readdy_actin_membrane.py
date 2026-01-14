import random

import numpy as np
from process_bigraph import Process
from simularium_readdy_models.actin import (
    ActinSimulation,
    ActinGenerator,
    FiberData,
)
from simularium_readdy_models.common import ReaddyUtil, get_membrane_monomers

def get_monomers():
    actin_monomers = ActinGenerator.get_monomers(
        fibers_data=[
            FiberData(
                28,
                [
                    np.array([-25, 0, 0]),
                    np.array([25, 0, 0]),
                ],
                "Actin-Polymer",
            )
        ],
        use_uuids=False,
        start_normal=np.array([0., 1., 0.]),
        longitudinal_bonds=True,
        barbed_binding_site=True,
    )
    actin_monomers = ActinGenerator.setup_fixed_monomers(
        actin_monomers,
        orthogonal_seed=True,
        n_fixed_monomers_pointed=3,
        n_fixed_monomers_barbed=0,
    )
    membrane_monomers = get_membrane_monomers(
        center=np.array([25.0, 0.0, 0.0]),
        size=np.array([0.0, 100.0, 100.0]),
        particle_radius=2.5,
        start_particle_id=len(actin_monomers["particles"].keys()),
        top_id=1
    )
    free_actin_monomers = ActinGenerator.get_free_actin_monomers(
        concentration=500.0,
        box_center=np.array([12., 0., 0.]),
        box_size=np.array([20., 50., 50.]),
        start_particle_id=len(actin_monomers["particles"].keys()) + len(membrane_monomers["particles"].keys()),
        start_top_id=2
    )
    monomers = {
        'particles': {**actin_monomers['particles'], **membrane_monomers['particles']},
        'topologies': {**actin_monomers['topologies'], **membrane_monomers['topologies']}
    }
    monomers = {
        'particles': {**monomers['particles'], **free_actin_monomers['particles']},
        'topologies': {**monomers['topologies'], **free_actin_monomers['topologies']}
    }
    return monomers



class ReaddyActinMembrane(Process):
    '''
    This process runs ReaDDy models with coarse-grained particle 
    actin filaments and membrane patches.
    '''

    config_schema = {
        'name': 'string{actin_membrane}',
        'internal_timestep': 'float{0.1}',
        'box_size': 'tuple[float,float,float]{(150.0, 150.0, 150.0)}',
        'periodic_boundary': 'boolean{true}',
        'reaction_distance': 'float{1.0}',
        'n_cpu': 'integer{4}',
        'only_linear_actin_constraints': 'boolean{true}',
        'reactions': 'boolean{true}',
        'dimerize_rate': 'float{1e-30}',
        'dimerize_reverse_rate': 'float{1.4e-9}',
        'trimerize_rate': 'float{2.1e-2}',
        'trimerize_reverse_rate': 'float{1.4e-9}',
        'pointed_growth_ATP_rate': 'float{2.4e-5}',
        'pointed_growth_ADP_rate': 'float{2.95e-6}',
        'pointed_shrink_ATP_rate': 'float{8.0e-10}',
        'pointed_shrink_ADP_rate': 'float{3.0e-10}',
        'barbed_growth_ATP_rate': 'float{1e30}',
        'barbed_growth_ADP_rate': 'float{7.0e-5}',
        'nucleate_ATP_rate': 'float{2.1e-2}',
        'nucleate_ADP_rate': 'float{7.0e-5}',
        'barbed_shrink_ATP_rate': 'float{1.4e-9}',
        'barbed_shrink_ADP_rate': 'float{8.0e-9}',
        'arp_bind_ATP_rate': 'float{2.1e-2}',
        'arp_bind_ADP_rate': 'float{7.0e-5}',
        'arp_unbind_ATP_rate': 'float{1.4e-9}',
        'arp_unbind_ADP_rate': 'float{8.0e-9}',
        'barbed_growth_branch_ATP_rate': 'float{2.1e-2}',
        'barbed_growth_branch_ADP_rate': 'float{7.0e-5}',
        'debranching_ATP_rate': 'float{1.4e-9}',
        'debranching_ADP_rate': 'float{7.0e-5}',
        'cap_bind_rate': 'float{2.1e-2}',
        'cap_unbind_rate': 'float{1.4e-9}',
        'hydrolysis_actin_rate': 'float{1e-30}',
        'hydrolysis_arp_rate': 'float{3.5e-5}',
        'nucleotide_exchange_actin_rate': 'float{1e-5}',
        'nucleotide_exchange_arp_rate': 'float{1e-5}',
        'verbose': 'boolean{false}',
        'use_box_actin': 'boolean{true}',
        'use_box_arp': 'boolean{false}',
        'use_box_cap': 'boolean{false}',
        'obstacle_radius': 'float{0.0}',
        'obstacle_diff_coeff': 'float{0.0}',
        'use_box_obstacle': 'boolean{false}',
        'position_obstacle_stride': 'integer{0}',
        'displace_pointed_end_tangent': 'boolean{false}',
        'displace_pointed_end_radial': 'boolean{false}',
        'tangent_displacement_nm': 'float{0.0}',
        'radial_displacement_radius_nm': 'float{0.0}',
        'radial_displacement_angle_deg': 'float{0.0}',
        'longitudinal_bonds': 'boolean{true}',
        'displace_stride': 'integer{1}',
        'bonds_force_multiplier': 'float{0.2}',
        'angles_force_constant': 'float{1000.0}',
        'dihedrals_force_constant': 'float{1000.0}',
        'actin_constraints': 'boolean{true}',
        'use_box_actin': 'boolean{true}',
        'actin_box_center_x': 'float{12.0}',
        'actin_box_center_y': 'float{0.0}',
        'actin_box_center_z': 'float{0.0}',
        'actin_box_size_x': 'float{20.0}',
        'actin_box_size_y': 'float{50.0}',
        'actin_box_size_z': 'float{50.0}',
        'add_extra_box': 'boolean{false}',
        'barbed_binding_site': 'boolean{true}',
        'binding_site_reaction_distance': 'float{3.0}',
        'add_membrane': 'boolean{true}',
        "membrane_center_x": 'float{25.0}',
        "membrane_center_y": 'float{0.0}',
        "membrane_center_z": 'float{0.0}',
        "membrane_size_x": 'float{0.0}',
        "membrane_size_y": 'float{100.0}',
        "membrane_size_z": 'float{100.0}',
        'membrane_particle_radius': 'float{2.5}',
        'obstacle_controlled_position_x': 'float{0.0}',
        'obstacle_controlled_position_y': 'float{0.0}',
        'obstacle_controlled_position_z': 'float{0.0}',
        'random_seed': 'integer{0}'
    }

    def initialize(self, config, readdy_system=None):
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        actin_simulation = ActinSimulation(self.config, False, False, readdy_system)
        self.readdy_system = actin_simulation.system
        self.readdy_simulation = actin_simulation.simulation

        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])


    def initial_state(self):
        return get_monomers()


    def inputs(self):
        return {
            'topologies': 'map[topology]',
            'particles': 'map[particle]',
        }

    def outputs(self):
        return {
            'topologies': 'map[topology]',
            'particles': 'map[particle]',
        }

    def update(self, inputs, interval):

        self.initialize(self.config, self.readdy_system)

        ReaddyUtil.add_monomers_from_data(self.readdy_simulation, inputs)

        simulate_readdy(
            self.config["internal_timestep"], 
            self.readdy_system, 
            self.readdy_simulation, 
            interval
        )

        id_diff = id_difference(self.readdy_simulation.current_topologies)

        readdy_monomers = ReaddyUtil.get_current_monomers(
            self.readdy_simulation.current_topologies, id_diff
        )

        return readdy_monomers


def id_difference(current_topologies):
    """
    Get the first ID coming out of ReaDDy, it should be zero 
    unless Readdy ran multiple times and cached the IDs, which
    would cause Vivarium to create new particles instead of 
    updating existing particles. 

    (This is a HACK needed as long as ReaDDy has this behavior.)
    """
    return current_topologies[0].particles[0].id


def simulate_readdy(internal_timestep, readdy_system, readdy_simulation, timestep):
    """
    Simulate in ReaDDy for the given timestep
    """
    def loop():
        readdy_actions = readdy_simulation._actions
        init = readdy_actions.initialize_kernel()
        diffuse = readdy_actions.integrator_euler_brownian_dynamics(
            internal_timestep
        )
        calculate_forces = readdy_actions.calculate_forces()
        create_nl = readdy_actions.create_neighbor_list(
            readdy_system.calculate_max_cutoff().magnitude
        )
        update_nl = readdy_actions.update_neighbor_list()
        react = readdy_actions.reaction_handler_uncontrolled_approximation(
            internal_timestep
        )
        init()
        create_nl()
        calculate_forces()
        update_nl()
        n_steps = int(timestep / internal_timestep)
        print(f"running readdy for {n_steps} steps")
        for t in range(1, n_steps + 1):
            diffuse()
            update_nl()
            react()
            update_nl()
            calculate_forces()

    readdy_simulation._run_custom_loop(loop, show_summary=False)

