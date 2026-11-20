from pathlib import Path as _Path
import yaml as _yaml

with _Path(__file__).with_name('config.yaml').open() as _config_file:
    _config = _yaml.safe_load(_config_file)
if (not isinstance(_config, dict) or set(_config) != {'sumo', 'max_episodes', 'evaluation'}
        or not isinstance(_config['sumo'], dict) or set(_config['sumo']) != {'backend'}):
    raise ValueError('config.yaml must contain max_episodes, sumo.backend and evaluation')
_max_episodes = _config['max_episodes']
if type(_max_episodes) is not int or _max_episodes < 1:
    raise ValueError('max_episodes must be a positive integer')
_sumo_backend = _config['sumo']['backend']
if _sumo_backend not in ('traci', 'libsumo'):
    raise ValueError('sumo.backend must be traci or libsumo')

_evaluation = _config['evaluation']
if not isinstance(_evaluation, dict) or set(_evaluation) != {
        'algorithm', 'checkpoint', 'network', 'seeds', 'policy_seed', 'max_steps'}:
    raise ValueError('evaluation must contain algorithm, checkpoint, network, seeds, policy_seed and max_steps')
if _evaluation['algorithm'] not in ('Unicorn', 'HeteroLight', 'FixedTime', 'Greedy', 'MaxPressure'):
    raise ValueError('Unknown evaluation.algorithm')
if _evaluation['checkpoint'] is not None and not isinstance(_evaluation['checkpoint'], str):
    raise ValueError('evaluation.checkpoint must be a path string or null')
if (not isinstance(_evaluation['network'], str)
        or _Path(_evaluation['network']).name != _evaluation['network']
        or not (_Path(__file__).parent / 'maps' / _evaluation['network'] / f"{_evaluation['network']}_config.json").is_file()):
    raise ValueError('evaluation.network must name an existing maps directory')
if (not isinstance(_evaluation['seeds'], list) or not _evaluation['seeds']
        or any(type(s) is not int or s < 0 or s >= 2**31 for s in _evaluation['seeds'])
        or len(set(_evaluation['seeds'])) != len(_evaluation['seeds'])):
    raise ValueError('evaluation.seeds must be a nonempty list of distinct nonnegative integer seeds below 2**31')
if type(_evaluation['policy_seed']) is not int or not 0 <= _evaluation['policy_seed'] < 2**32:
    raise ValueError('evaluation.policy_seed must be an integer in [0, 2**32)')
if type(_evaluation['max_steps']) is not int or _evaluation['max_steps'] <= 0:
    raise ValueError('evaluation.max_steps must be a positive integer')


class INPUT_PARAMS:
    MAX_EPISODES        = _max_episodes
    NUM_META_AGENTS     = 6
    OBS_TYPE            = 'V1'  # ['V1', 'V2']
    UPDATE_TYPE         = 'PPO_MAC'
    LOAD_MODEL          = False
    EXPERIMENT_PATH     = None
    CO_TRAIN            = False


class EVALUATION_PARAMS:
    ALGORITHM           = _evaluation['algorithm']
    CHECKPOINT          = _evaluation['checkpoint']
    NETWORK             = _evaluation['network']
    SEEDS               = _evaluation['seeds']
    POLICY_SEED         = _evaluation['policy_seed']
    MAX_STEPS           = _evaluation['max_steps']


class SUMO_PARAMS:
    BACKEND             = _sumo_backend
    # MA2C DATASETS: 'grid_network_5_5', 'monaco_network_30',
    # RESCO DATASETS: 'cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4', 'grid_network_4_4'
    # GESA DATASETS: 'shaoxing_network_7', 'shenzhen_network_29', 'shenzhen_network_55'
    ALL_DATASETS        = ['cologne_network_8', 'ingolstadt_network_21', 'arterial_network_4_4',
                           'grid_network_4_4', 'shaoxing_network_7', 'shenzhen_network_29']
    NET_NAME            = 'grid_network_5_5' # ['grid_network_5_5', 'monaco_network_30']
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    GUI                 = False
    RANDOM_SEED         = True
    OBS_SHARING         = False
    REWARD_DETECTOR     = True
    REGIONAL_REWARD     = True
    MAX_DISTANCE        = None
    SEED                = 1
    MAX_SUMO_STEP       = 3600
    # MA2C: 5s/2s, RESCO and SG: 10s/3s (5s/2s for arterial4x4), GESA: 10s/5s
    GREEN_DURATION      = 15
    YELLOW_DURATION     = 5
    TELEPORT_TIME       = 300  # (300 for ma2c and sg dataset, 600 for GESA dataset, -1 for RESCO dataset)
    # random route
    END_TIME            = 300
    PERIOD              = 2
    MIN_DIS             = 200
    FRINGE_FACTOR       = 3
    # baseline route for grid 5x5 grid network (default: 1100, 925)
    PEAK_FLOW1          = 1100
    PEAK_FLOW2          = 925
    INIT_DENSITY        = 0
    # baseline route for 30 monaco network (default: 325)
    FLOW                = 325
    # baseline route for 16 singapore network
    SG_FLOW             = 1100


class TRAIN_PARAMS:
    MAX_EPISODE         = INPUT_PARAMS.MAX_EPISODES
    NUM_META_AGENTS     = INPUT_PARAMS.NUM_META_AGENTS
    LOAD_MODEL          = INPUT_PARAMS.LOAD_MODEL
    EXPERIMENT_PATH     = INPUT_PARAMS.EXPERIMENT_PATH
    CONTRASTIVE_LOSS    = False
    NEIGHBOR_REWARD     = False
    WANDB               = False
    USE_GPU             = True
    NUM_GPU             = 1
    SAVE_MODEL_STEP     = 500
    SUMMARY_WINDOW      = 30
    RESET_OPTIM         = False
    RESET_OPTIM_STEP    = 20000


class NETWORK_PARAMS:
    CO_TRAIN            = INPUT_PARAMS.CO_TRAIN
    OBS_TYPE            = INPUT_PARAMS.OBS_TYPE
    UPDATE_TYPE         = INPUT_PARAMS.UPDATE_TYPE
    VL_FACTOR           = 0.5
    EL_FACTOR           = 2e-3
    PL_FACTOR           = 2e-4
    CL_FACTOR           = 1e-5
    CONTRASTIVE_TEMP    = 0.2
    CONTRASTIVE_BATCH   = 256
    GAMMA               = .95
    LAMBDA_ADV          = .98
    A_LR_Q              = 1.e-4
    C_LR_Q              = 2.e-4
    GRAD_CLIP           = 10
    EPS_CLIP            = 0.2
    K_EPOCH             = 6


class JOB_OPTIONS:
    GET_EXPERIENCE       = 1
    GET_GRADIENT         = 2


class COMPUTE_OPTIONS:
    MULTI_THREADED       = 1
    SINGLE_THREADED      = 2


class ALGORITHM_OPTIONS:
    A3C                  = 1
    PPO                  = 2


JOB_TYPE                 = JOB_OPTIONS.GET_EXPERIENCE
COMPUTE_TYPE             = COMPUTE_OPTIONS.SINGLE_THREADED
ALGORITHM_TYPE           = ALGORITHM_OPTIONS.PPO
