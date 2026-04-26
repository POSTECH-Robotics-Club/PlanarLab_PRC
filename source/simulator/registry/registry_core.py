from typing import Dict

TASK_REGISTRY: Dict[str, dict] = {}


def register_task(name: str):
    def decorator(cls):
        if name in TASK_REGISTRY:
            raise ValueError(f"{name} already registered")

        if hasattr(cls, "spec"):
            TASK_REGISTRY[name] = cls.spec()
        else:
            raise ValueError(f"{name} must define spec()")

        return cls

    return decorator
