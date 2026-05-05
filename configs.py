import os
import multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Worker count for multiprocessing pool. Defaults to cpu_count - 2 so the
# main process and OS stay responsive. Override via env var for cluster runs:
#   N_WORKERS=16 python run.py
N_WORKERS = int(os.environ.get('N_WORKERS', max(1, multiprocessing.cpu_count() - 2)))

# Paths
RESULTS_DIR = './data/results/'
FIGURES_DIR = './data/figures/'
PLOTS_DIR = './data/plots/'
TESTS_PLOTS_DIR = './data/tests_plots/'
TESTS_VIDEOS_DIR = './data/tests_videos/'
FIG_FMT = '.svg'

# MuJoCo XML asset directory (vendored in agents/xmls/ by default).
# Override via env var to use a different set of MJCF models:
#   MUJOCO_XML_DIR=/path/to/xmls python run.py
MUJOCO_XML_DIR = os.environ.get(
    'MUJOCO_XML_DIR',
    os.path.join(os.path.dirname(__file__), 'agents', 'xmls'),
)

# Shared simulation parameters
# DT is the default control timestep (50 Hz, matches Gymnasium).
# ENV_DT is the per-env lookup; each env runs at its MJPC agent_timestep.
DT = 0.02
ENV_DT = {'cartpole': 0.02, 'cartpole_quadratic': 0.02, 'walker': 0.01,
          'humanoid_stand': 0.015, 'humanoid_balance': 0.015,
          'humanoid_stand_gravity': 0.015}
SEED = 42
FAILURE_ANGLE = 30  # pole failure threshold [degrees]
POSITION_BOUND = 2.4  # cart position boundary [m]

# InvertedPendulum (MJX) constants
IP_FAILURE_ANGLE = 0.2   # rad — pole failure threshold, matches Gymnasium InvertedPendulum-v5
IP_CART_BOUND    = 1.0   # m — slider joint range in inverted_pendulum.xml

# Figure composition
SCALE_TEXT = 1.5
FIG_WIDTH = 6.27
FIG_CLEANUP = True
