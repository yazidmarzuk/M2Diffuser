from pathlib import Path

ASSETS_VERSION = "v0.0.6"

class RootPath:
    ENV_PATH = Path("/data1/sixuyan/m2diffuser/env_model")
    AGENT = ENV_PATH / "agent"
    SCENE = ENV_PATH / "physcene"