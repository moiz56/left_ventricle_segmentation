from pathlib import Path
import yaml

def get_yaml():
    try:

        PROJECT_ROOT = Path(__file__).resolve().parents[2]  
        CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"

        if not CONFIG_FILE.exists():
            raise FileNotFoundError(f"Config file not found at: {CONFIG_FILE}")

        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing yaml file:{e}")




