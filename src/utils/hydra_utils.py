from hydra import compose, core, initialize
from hydra.core.global_hydra import GlobalHydra

def setup_config(config_path):
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()
    # Initialize Hydra and load config manually
    initialize(config_path=config_path, version_base=None)  # Set path to your configs
    # Load the configuration
    cfg = compose(config_name="config")
    return cfg