import importlib
import inspect
import os
from .interface import Optimizer

def load_plugins(plugin_dirs=None):
    """
    Загружает все классы, наследующие Optimizer, из указанных директорий.
    """
    plugins = []

    def import_module_from_file(filepath):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    dirs_to_scan = []
    pkg_dir = os.path.dirname(__file__)
    dirs_to_scan.append(pkg_dir)

    if plugin_dirs is None:
        root_plugins = os.path.join(os.path.dirname(pkg_dir), 'plugins')
        if os.path.isdir(root_plugins):
            dirs_to_scan.append(root_plugins)
    else:
        dirs_to_scan.extend(plugin_dirs)

    for plugin_dir in dirs_to_scan:
        if not os.path.isdir(plugin_dir):
            continue
        for filename in os.listdir(plugin_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                filepath = os.path.join(plugin_dir, filename)
                try:
                    module = import_module_from_file(filepath)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Optimizer) and obj is not Optimizer:
                            plugins.append(obj)
                except Exception as e:
                    print(f"Ошибка загрузки плагина оптимизатора {filepath}: {e}")

    return plugins

def load_builtin_plugins():
    plugins = []
    for importer, modname, ispkg in pkgutil.iter_modules(__path__):
        if modname == 'interface':
            continue
        module = importlib.import_module(f'.{modname}', package=__name__)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Optimizer) and obj is not Optimizer:
                plugins.append(obj)
    return plugins

from .grid_search import GridSearchOptimizer