from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def load_plugin_class(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    plugin_class = getattr(module, class_name)
    if not isinstance(plugin_class, type):
        raise TypeError(f"{class_name} in {module_name} is not a class.")
    return plugin_class


def load_plugin_class_from_file(file_path: str | Path, class_name: str) -> type:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Plugin file not found: {path}")

    module_name = f"dynamic_plugin_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin_class = getattr(module, class_name)
    if not isinstance(plugin_class, type):
        raise TypeError(f"{class_name} in {path} is not a class.")
    return plugin_class


def create_plugin(plugin_ref: str, **kwargs):
    if ":" not in plugin_ref:
        raise ValueError("plugin_ref must be in format '<module_or_file>:<class_name>'")

    module_or_file, class_name = plugin_ref.split(":", 1)
    if module_or_file.endswith(".py") or "/" in module_or_file:
        plugin_class = load_plugin_class_from_file(module_or_file, class_name)
    else:
        plugin_class = load_plugin_class(module_or_file, class_name)
    return plugin_class(**kwargs)
