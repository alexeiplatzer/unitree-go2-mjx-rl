from dataclasses import dataclass


@dataclass
class BaseEnvConfig:
    backend: str = "mjx"
