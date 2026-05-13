import ast
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class ProjectStructureScanner:
    """
    Сканирует python-проект и строит читаемое текстовое дерево:
    - структура директорий и файлов
    - классы, dataclasses, enums, TypedDict
    - методы (с сигнатурами, async, @classmethod/@staticmethod)
    - top-level функции
    """

    def __init__(self, ignore_dirs: Optional[List[str]] = None):
        self.ignore_dirs = ignore_dirs or [
            '__pycache__', '.git', '.idea', 'venv', 'env', '.venv', 'node_modules', 'dist', 'build'
        ]
        self.ignore_files = {'__init__.py'}

    def scan(self, path: Union[str, Path], lang: str = "python") -> str:
        if lang.lower() != "python":
            raise ValueError("Пока поддерживается только lang='python'")

        root = Path(path).resolve()
        if not root.is_dir():
            raise ValueError(f"Путь не является директорией: {root}")

        lines = [f"{root.name}/"]
        self._walk_directory(root, root, lines, indent="")
        return "\n".join(lines)

    def _walk_directory(
        self,
        root: Path,
        current: Path,
        lines: List[str],
        indent: str = ""
    ):
        items = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))

        for idx, item in enumerate(items):
            is_last = idx == len(items) - 1
            prefix = "└── " if is_last else "├── "
            next_indent = indent + ("    " if is_last else "│   ")

            if item.is_dir():
                if item.name in self.ignore_dirs:
                    continue
                lines.append(f"{indent}{prefix}{item.name}/")
                self._walk_directory(root, item, lines, next_indent)

            elif item.is_file() and item.suffix == ".py" and item.name not in self.ignore_files:
                lines.append(f"{indent}{prefix}{item.name}")

                try:
                    symbols = self._extract_symbols(item)
                    if symbols:
                        file_indent = next_indent
                        for kind, name, members in symbols:
                            sym_prefix = "    " if is_last else "│   "
                            lines.append(f"{file_indent}{sym_prefix}└─ {kind} {name}")
                            if members:
                                member_indent = file_indent + ("    " if is_last else "│   ")
                                for member in members:
                                    lines.append(f"{member_indent}   ├─ {member}")
                except Exception as e:
                    lines.append(f"{next_indent}    # (parse error: {str(e)})")

    def _extract_symbols(self, filepath: Path) -> List[Tuple[str, str, List[str]]]:
        """
        Возвращает список кортежей:
        [("class", "ClassName", ["method1(...)", "method2(...)"]),
         ("dataclass", "UserData", ["id: int", "name: str", "greet()"]),
         ("def", "top_level_func(arg: int) -> str", [])]
        """
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        symbols = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                decorators = [self._get_decorator_name(d) for d in node.decorator_list]

                if "dataclass" in decorators:
                    kind = "dataclass"
                elif any("Enum" in d for d in decorators) or node.bases and any("Enum" in self._format_annotation(b) for b in node.bases):
                    kind = "enum"
                elif "TypedDict" in decorators or any("TypedDict" in d for d in decorators):
                    kind = "TypedDict"
                else:
                    kind = "class"

                members = []

                for subnode in node.body:
                    if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if subnode.name.startswith('_'):
                            continue  # скрываем приватные и защищённые методы
                        sig = self._get_function_signature(subnode)
                        members.append(sig)

                    elif isinstance(subnode, ast.AnnAssign) and isinstance(subnode.target, ast.Name):
                        # поля dataclass / TypedDict
                        name = subnode.target.id
                        ann = self._format_annotation(subnode.annotation) if subnode.annotation else "Any"
                        members.append(f"{name}: {ann}")

                    elif isinstance(subnode, ast.Assign) and len(subnode.targets) == 1:
                        if isinstance(subnode.targets[0], ast.Name):
                            members.append(f"{subnode.targets[0].id} = ...")

                symbols.append((kind, node.name, sorted(members)))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # top-level функции
                if node.name.startswith('_'):
                    continue
                sig = self._get_function_signature(node)
                symbols.append(("def", sig, []))

        return sorted(symbols, key=lambda x: (x[0] == "def", x[1].lower()))

    def _get_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        prefix = ""
        if isinstance(node, ast.AsyncFunctionDef):
            prefix = "async "
        if "classmethod" in decorators:
            prefix += "@classmethod "
        elif "staticmethod" in decorators:
            prefix += "@staticmethod "

        name = node.name

        args_str = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._format_annotation(arg.annotation)}"
            args_str.append(arg_str)

        if node.args.vararg:
            args_str.append(f"*{node.args.vararg.arg}")

        if node.args.kwarg:
            args_str.append(f"**{node.args.kwarg.arg}")

        args_part = ", ".join(args_str) if args_str else ""

        return_type = ""
        if node.returns:
            return_type = f" → {self._format_annotation(node.returns)}"

        return f"{prefix}{name}({args_part}){return_type}"

    def _get_decorator_name(self, dec) -> str:
        if isinstance(dec, ast.Name):
            return dec.id
        if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
            return f"{dec.value.id}.{dec.attr}"
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            return dec.func.id
        return "?"

    def _format_annotation(self, ann) -> str:
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant):
            return repr(ann.value)
        if isinstance(ann, ast.Subscript):
            value = self._format_annotation(ann.value)
            slice_val = self._format_annotation(ann.slice) if hasattr(ann, 'slice') else "?"
            return f"{value}[{slice_val}]"
        if isinstance(ann, ast.Attribute):
            return f"{self._format_annotation(ann.value)}.{ann.attr}"
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            # Python 3.10+ union types: int | str
            left = self._format_annotation(ann.left)
            right = self._format_annotation(ann.right)
            return f"{left} | {right}"
        # fallback
        try:
            return ast.unparse(ann)
        except Exception:
            return "???"


# ────────────────────────────────────────────────
if __name__ == "__main__":
    scanner = ProjectStructureScanner()
    try:
        structure = scanner.scan(".", lang="python")
        logger.info(structure)
    except Exception as e:
        logger.error(f"Ошибка: {e}")