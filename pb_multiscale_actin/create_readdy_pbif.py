import random
from typing import Any

import numpy as np
from process_bigraph import ProcessTypes, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results
from simularium_readdy_models.actin import ActinGenerator, FiberData
from simularium_readdy_models.common import get_membrane_monomers

from pb_multiscale_actin.processes import ReaddyActinMembrane
from pb_multiscale_actin.processes import SimulariumEmitter



def register_items_into_core(core: ProcessTypes):
    particle = {
        'type_name': 'string',
        'position': 'tuple[float,float,float]',
        'neighbor_ids': 'list[integer]',
        '_apply': 'set',
    }
    topology = {
        'type_name': 'string',
        'particle_ids': 'list[integer]',
        '_apply': 'set',
    }
    core.register('topology', topology)
    core.register('particle', particle)

    core.register_process('pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane', ReaddyActinMembrane)
    core.register_process('pb_multiscale_actin.processes.readdy_actin_membrane.SimulariumEmitter', SimulariumEmitter)


def generate_readdy_pbg(output_dir: str):
    emitters_from_wires = emitter_from_wires({
        'particles': ['particles'],
        'topologies': ['topologies'],
        'global_time': ['global_time']
    }, address='local:pb_multiscale_actin.processes.readdy_actin_membrane.SimulariumEmitter')
    emitters_from_wires["config"]["output_dir"] = output_dir

    state = {
        "emitter": emitters_from_wires,
        'readdy': {
            '_type': 'process',
            'address': 'local:pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane',
            'inputs': {
                'particles': ['particles'],
                'topologies': ['topologies']
            },
            'outputs': {
                'particles': ['particles'],
                'topologies': ['topologies']
            }
        },
    }
    return state

def run_readdy_actin_membrane(total_time=3):
    state = generate_readdy_pbg(output_dir="readdy_result")

    core = ProcessTypes()
    register_items_into_core(core)

    sim = Composite({
        "state": state,
    }, core=core)

    # simulate
    sim.run(total_time)  # time in ns

    results = gather_emitter_results(sim)


if __name__ == "__main__":
    run_readdy_actin_membrane()


