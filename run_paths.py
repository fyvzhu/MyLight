"""Create one run in the parent process; pass its paths to every worker."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import re
import xml.etree.ElementTree as ET
import yaml
import parameters as settings

PROJECT_ROOT = Path(__file__).resolve().parent
LEARNING_ALGORITHMS = ('Unicorn', 'HeteroLight')
RULE_ALGORITHMS = ('FixedTime', 'Greedy', 'MaxPressure')


def project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _public(cls):
    return {k: v for k, v in vars(cls).items() if not k.startswith('_')}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    stage: str
    algorithm: str

    @classmethod
    def create(cls, stage, algorithm):
        if stage not in ('train', 'evaluate'):
            raise ValueError(f'Unknown stage: {stage}')
        allowed = LEARNING_ALGORITHMS if stage == 'train' else LEARNING_ALGORITHMS + RULE_ALGORITHMS
        if algorithm not in allowed:
            raise ValueError(f'{algorithm} is not a {stage} algorithm')
        parent = PROJECT_ROOT / ('train_outputs' if stage == 'train' else 'evaluate_outputs') / algorithm
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        root = parent / f'{algorithm}_{stamp}'
        root.mkdir(exist_ok=False)
        run = cls(root, stage, algorithm)
        children = (run.model, run.tensorboard) if stage == 'train' else (run.data,)
        for child in (*children, run.simulation):
            child.mkdir()
        return run

    @property
    def model(self):
        return self.root / 'model'

    @property
    def tensorboard(self):
        return self.root / 'tensorboard'

    @property
    def data(self):
        return self.root / 'data'

    @property
    def simulation(self):
        return self.root / 'simulation'

    def save_config(self, checkpoint=None):
        sumo = _public(settings.SUMO_PARAMS)
        if self.stage == 'evaluate':
            sumo['NET_NAME'] = settings.EVALUATION_PARAMS.NETWORK
        sumo['NET_PATH'] = str(PROJECT_ROOT / 'maps' / sumo['NET_NAME'])
        sumo['CONFIG_PATH'] = str(Path(sumo['NET_PATH']) / f"{sumo['NET_NAME']}_config.json")
        config = {'stage': self.stage, 'algorithm': self.algorithm, 'run_dir': str(self.root), 'sumo': sumo}
        if self.stage == 'train':
            config.update(max_episodes=settings.INPUT_PARAMS.MAX_EPISODES,
                          input=_public(settings.INPUT_PARAMS), training=_public(settings.TRAIN_PARAMS),
                          network=_public(settings.NETWORK_PARAMS))
        else:
            config['evaluation'] = _public(settings.EVALUATION_PARAMS)
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            hasher = hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
            config['checkpoint'] = {'path': str(checkpoint.resolve()), 'sha256': hasher.hexdigest()}
        with (self.root / 'config.yaml').open('w') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

    def workspace(self, worker, network):
        # Preserve scenario basenames expected by the original route generators.
        label = 'controller' if worker is None else f'worker_{worker}'
        return self.simulation / label / network


def prepare_simulation(source, workspace):
    """Reference static inputs and isolate generated routes, detector and XML outputs."""
    source, workspace = Path(source), Path(workspace)
    if workspace.resolve().is_relative_to(PROJECT_ROOT / 'maps'):
        raise ValueError('Simulation output must be outside the original maps directory')
    workspace.mkdir(parents=True, exist_ok=True)
    dynamic = source.name in ('grid_network_5_5', 'monaco_network_30')
    output_tags = {'laneAreaDetector', 'inductionLoop', 'e1Detector', 'e2Detector',
                   'e3Detector', 'entryExitDetector', 'instantInductionLoop'}
    for path in source.iterdir():
        name = path.name
        if (not path.is_file() or path.suffix not in ('.xml', '.json')
                or name.startswith(('trip_', 'statistic_', 'routes_output_'))
                or re.fullmatch(re.escape(source.name) + r'_\d+\.rou\.xml', name)
                or dynamic and name == source.name + '.rou.xml'):
            continue
        target = workspace / name
        if target.exists():
            continue
        if path.suffix == '.xml':
            tree = ET.parse(path)
            if tree.getroot().tag not in ('net', 'routes', 'additional', 'viewsettings'):
                continue
            outputs = [node for node in tree.getroot().iter() if node.tag in output_tags and 'file' in node.attrib]
            if outputs:
                for node in outputs:
                    value = node.get('file')
                    if value not in ('NUL', '/dev/null', 'stdout', 'stderr'):
                        node.set('file', str(workspace / Path(value).name))
                tree.write(target, encoding='utf-8', xml_declaration=True)
                continue
        target.symlink_to(path.resolve())


def confine_config_outputs(config_path):
    """Keep generator-provided outputs, such as Shaoxing FCD, in this workspace."""
    config_path = Path(config_path)
    tree = ET.parse(config_path)
    outputs = tree.getroot().find('output')
    if outputs is None:
        return
    changed = False
    for node in outputs:
        value = node.get('value')
        if node.tag.endswith('-output') and value and value not in ('NUL', '/dev/null', 'stdout', 'stderr'):
            node.set('value', str(config_path.parent / Path(value).name))
            changed = True
    if changed:
        tree.write(config_path, encoding='utf-8', xml_declaration=True)
