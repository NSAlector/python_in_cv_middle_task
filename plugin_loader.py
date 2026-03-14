import importlib
import os
import sys


class PluginLoader:
    def __init__(self, plugin_dirs=None):
        """
        Args:
            plugin_dirs: список директорий с плагинами
        """
        self.plugin_dirs = plugin_dirs or ['encoder', 'clusterizer', 'optimizer']
        self.plugins = {}

        for plugin_dir in self.plugin_dirs:
            if os.path.exists(plugin_dir) and plugin_dir not in sys.path:
                sys.path.insert(0, os.path.abspath(plugin_dir))

    def load_plugin(self, module_name, function_name='get_plugin'):
        """
        Загружает плагин из модуля
        """
        try:
            module = None
            for plugin_dir in self.plugin_dirs:
                module_path = f"{plugin_dir}.{module_name}"
                try:
                    module = importlib.import_module(module_path)
                    break
                except ImportError:
                    continue

            if module is None:
                module = importlib.import_module(module_name)

            get_func = getattr(module, function_name, None)
            if get_func is None:
                class_name = ''.join(word.capitalize() for word in module_name.split('_'))
                return getattr(module, class_name, None)

            return get_func()

        except Exception as e:
            print(f"Ошибка при загрузке плагина {module_name}: {e}")
            return None

    def load_encoder(self, name='encoder'):
        result = self.load_plugin(name, 'get_encoder')
        if result is None:
            print("Ошибка: Не удалось загрузить кодировщик")
        return result

    def load_clusterizer(self, name='clusterizer'):
        result = self.load_plugin(name, 'get_clusterizer')
        if result is None:
            print("Ошибка: Не удалось загрузить кластеризатор")
        return result

    def load_optimizer(self, name='optimizer'):
        result = self.load_plugin(name, 'get_optimizer')
        if result is None:
            print("Ошибка: Не удалось загрузить оптимизатор")
        return result
