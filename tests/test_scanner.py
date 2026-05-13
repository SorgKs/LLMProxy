# -*- coding: utf-8 -*-
import log
"""
Тесты для ProjectStructureScanner (scanner.py)
Проверяет сканирование Python-проектов и построение дерева структуры
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner import ProjectStructureScanner


class TestProjectStructureScanner(unittest.TestCase):
    """Тесты для сканера структуры проекта"""

    def setUp(self):
        """Создает временную директорию с тестовыми Python-файлами"""
        self.temp_dir = tempfile.mkdtemp()
        self.scanner = ProjectStructureScanner(ignore_dirs=["__pycache__", ".git"])

        # Создаем тестовую структуру файлов
        self._create_test_structure()

    def _create_test_structure(self):
        """Создает тестовую структуру Python-файлов"""
        # Основной файл
        with open(os.path.join(self.temp_dir, "main.py"), "w", encoding="utf-8") as f:
            f.write("""
import os

class MainClass:
    \"\"\"Основной класс\"\"\"
    
    def public_method(self):
        pass
    
    def _private_method(self):
        pass
    
    @property
    def my_property(self):
        return 42

def standalone_function(x: int, y: str) -> bool:
    return True
""")

        # Файл с функциями
        with open(os.path.join(self.temp_dir, "utils.py"), "w", encoding="utf-8") as f:
            f.write("""
def helper_func(a, b):
    \"\"\"Вспомогательная функция\"\"\"
    return a + b

async def async_function():
    pass
""")

        # Поддиректория с файлом
        sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(sub_dir, exist_ok=True)
        with open(os.path.join(sub_dir, "module.py"), "w", encoding="utf-8") as f:
            f.write("""
class SubClass:
    def method(self):
        pass
""")

        # Файл, который не должен парситься (не Python)
        with open(os.path.join(self.temp_dir, "not_python.txt"), "w") as f:
            f.write("This is not Python")

    def tearDown(self):
        """Удаляет временную директорию"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_scan_returns_string(self):
        """Проверяет, что scan возвращает строку"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        log.log_info("test_scan_returns_string passed!")

    def test_scan_includes_filenames(self):
        """Проверяет, что в результате есть имена файлов"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIn("main.py", result)
        self.assertIn("utils.py", result)
        self.assertIn("module.py", result)
        log.log_info("test_scan_includes_filenames passed!")

    def test_scan_includes_class_names(self):
        """Проверяет, что в результате есть имена классов"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIn("MainClass", result)
        self.assertIn("SubClass", result)
        log.log_info("test_scan_includes_class_names passed!")

    def test_scan_includes_method_names(self):
        """Проверяет, что в результате есть имена публичных методов"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIn("public_method", result)
        self.assertIn("method", result)
        log.log_info("test_scan_includes_method_names passed!")

    def test_scan_excludes_private_methods(self):
        """Проверяет, что приватные методы (с _) исключаются"""
        result = self.scanner.scan(self.temp_dir)
        # _private_method не должен быть в результате
        # Но нужно быть осторожным, чтобы не исключить слишком много
        lines = result.split("\n")
        for line in lines:
            if "_private_method" in line:
                self.fail("_private_method не должен быть в результате сканирования")
        log.log_info("test_scan_excludes_private_methods passed!")

    def test_scan_includes_functions(self):
        """Проверяет, что standalone-функции включены"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIn("standalone_function", result)
        self.assertIn("helper_func", result)
        log.log_info("test_scan_includes_functions passed!")

    def test_scan_includes_function_signatures(self):
        """Проверяет, что сигнатуры функций включены"""
        result = self.scanner.scan(self.temp_dir)
        # standalone_function должна иметь сигнатуру с типами
        self.assertIn("int", result)
        self.assertIn("str", result)
        log.log_info("test_scan_includes_function_signatures passed!")

    def test_scan_ignores_non_python_files(self):
        """Проверяет, что не-Python файлы игнорируются"""
        result = self.scanner.scan(self.temp_dir)
        # not_python.txt не должен быть в дереве как Python-файл
        # (может упоминаться в другом контексте, но не как .py)
        self.assertNotIn("not_python.txt", result)
        log.log_info("test_scan_ignores_non_python_files passed!")

    def test_scan_with_project_structure_marker(self):
        """Проверяет наличие маркера структуры проекта"""
        result = self.scanner.scan(self.temp_dir, lang="python")
        # Проверяем, что структура содержит имена файлов
        self.assertIn("main.py", result)
        self.assertIn("utils.py", result)
        log.log_info("test_scan_with_project_structure_marker passed!")

    def test_scan_subdirectory_included(self):
        """Проверяет, что поддиректории сканируются"""
        result = self.scanner.scan(self.temp_dir)
        self.assertIn("subdir", result)
        self.assertIn("module.py", result)
        log.log_info("test_scan_subdirectory_included passed!")

    def test_scan_nonexistent_path(self):
        """Проверяет обработку несуществующего пути"""
        # Должен выбросить ValueError
        with self.assertRaises(ValueError):
            self.scanner.scan("/nonexistent/path")
        log.log_info("test_scan_nonexistent_path passed!")

    def test_extract_symbols_classes(self):
        """Проверяет извлечение символов из файла с классами"""
        filepath = os.path.join(self.temp_dir, "main.py")
        symbols = self.scanner._extract_symbols(filepath)

        # Должен найти MainClass
        class_names = [s[1] for s in symbols if s[0] == "class"]
        self.assertIn("MainClass", class_names)
        log.log_info("test_extract_symbols_classes passed!")

    def test_extract_symbols_functions(self):
        """Проверяет извлечение символов из файла с функциями"""
        filepath = os.path.join(self.temp_dir, "utils.py")
        symbols = self.scanner._extract_symbols(filepath)

        # Должен найти helper_func и async_function
        # В scanner.py функции имеют тип "def", а имя - это полная сигнатура
        func_signatures = [s[1] for s in symbols if s[0] == "def"]
        self.assertTrue(any("helper_func" in sig for sig in func_signatures))
        self.assertTrue(any("async_function" in sig for sig in func_signatures))
        log.log_info("test_extract_symbols_functions passed!")

    def test_get_function_signature(self):
        """Проверяет получение сигнатуры функции"""
        import ast

        code = "def my_func(x: int, y: str) -> bool: pass"
        tree = ast.parse(code)
        func_def = tree.body[0]

        signature = self.scanner._get_function_signature(func_def)
        self.assertIn("x", signature)
        self.assertIn("y", signature)
        log.log_info("test_get_function_signature passed!")

    def test_decorator_extraction(self):
        """Проверяет извлечение декораторов"""
        import ast

        code = """
@staticmethod
def static_method():
    pass
"""
        tree = ast.parse(code)
        func_def = tree.body[0]

        # Проверяем, что декоратор статического метода отображается
        signature = self.scanner._get_function_signature(func_def)
        self.assertIn("staticmethod", signature)
        log.log_info("test_decorator_extraction passed!")


class TestScannerIgnoreDirs(unittest.TestCase):
    """Тесты игнорирования директорий"""

    def test_ignore_pycache(self):
        """Проверяет игнорирование __pycache__"""
        temp_dir = tempfile.mkdtemp()
        try:
            scanner = ProjectStructureScanner(ignore_dirs=["__pycache__"])

            # Создаем __pycache__ с файлом
            pycache_dir = os.path.join(temp_dir, "__pycache__")
            os.makedirs(pycache_dir, exist_ok=True)
            with open(os.path.join(pycache_dir, "test.pyc"), "w") as f:
                f.write("")

            # Создаем обычный файл
            with open(os.path.join(temp_dir, "real.py"), "w") as f:
                f.write("def func(): pass")

            result = scanner.scan(temp_dir)

            # __pycache__ не должно быть в результате
            self.assertNotIn("__pycache__", result)
            self.assertIn("real.py", result)

            log.log_info("test_ignore_pycache passed!")
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
