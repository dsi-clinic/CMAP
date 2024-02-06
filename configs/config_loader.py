import argparse
import importlib


def load_config(module_name):
    """
    Load the configuration module dynamically based on the module name.

    Parameters
    ----------
    module_name : str
        The name of the module to load.

    Returns
    -------
    module
        The loaded configuration module.
    """
    config_module = importlib.import_module(module_name)
    return config_module


def main():
    parser = argparse.ArgumentParser(description="Load dynamic config")
    parser.add_argument("config", type=str, help="Config module name")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Data directory: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
