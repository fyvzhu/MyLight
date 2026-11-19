"""One simulator interface and per-timestep read cache for MATSC environments."""
import importlib
import re


class _Domain:
    def __init__(self, domain, dynamic, static, methods):
        self._domain = domain
        for name in methods:
            original = getattr(domain, name)
            storage = static if name == 'getIDList' else dynamic

            def read(*args, _original=original, _storage=storage, _name=name, **kwargs):
                key = (id(domain), _name, args, tuple(sorted(kwargs.items())))
                if key not in _storage:
                    _storage[key] = _original(*args, **kwargs)
                return _storage[key]

            setattr(self, name, read)

    def __getattr__(self, name):
        return getattr(self._domain, name)


class Simulator:
    """A process owns one SUMO simulation, using the backend chosen in config.yaml.

    Only read-only values are cached. Vehicle/detector values expire at every
    simulation step; the detector ID list expires on start/load/close. There is
    no implicit backend fallback and no change to detector or file outputs.
    """
    def __init__(self, backend):
        if backend not in ('traci', 'libsumo'):
            raise ValueError(f'Unknown SUMO backend: {backend}')
        self.backend = backend
        self._api = importlib.import_module(backend)
        self._dynamic = {}
        self._static = {}
        for name, methods in {
            'lanearea': ['getIDList', 'getLastStepHaltingNumber',
                         'getLastStepVehicleNumber', 'getLastStepOccupancy'],
            'lane': ['getLastStepVehicleIDs', 'getLastStepVehicleNumber', 'getLastStepHaltingNumber'],
            'vehicle': ['getSpeed', 'getLength', 'getLanePosition'],
        }.items():
            setattr(self, name, _Domain(getattr(self._api, name),
                                       self._dynamic, self._static, methods))

    def __getattr__(self, name):
        return getattr(self._api, name)

    def _clear(self):
        self._dynamic.clear()
        self._static.clear()

    def start(self, command, port=None):
        self._clear()
        if self.backend == 'libsumo':
            result = self._api.start(command)
        else:
            result = self._api.start(command, port=port)
        version = self._api.getVersion()[1]
        if not re.search(r'\b1\.16\.0\b', version):
            self.close()
            raise RuntimeError(f'SUMO 1.16.0 is required; {self.backend} reported {version}')
        return result

    def simulationStep(self, step=0):
        self._dynamic.clear()
        return self._api.simulationStep(step)

    def load(self, args):
        self._clear()
        return self._api.load(args)

    def close(self):
        try:
            return self._api.close()
        finally:
            self._clear()
