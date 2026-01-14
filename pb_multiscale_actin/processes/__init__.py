
from pb_multiscale_actin.processes.readdy_actin_membrane import ReaddyActinMembrane
from pb_multiscale_actin.processes.simularium_emitter import SimulariumEmitter


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

