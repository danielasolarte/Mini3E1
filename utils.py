import json
import os
from glob import glob


def converter(name: str) -> str:
    """Convierte el notebook `name`.ipynb en un script `name`.py.

    Solo se copian las celdas de código; los comandos mágicos y de shell
    (líneas que empiezan por '%', '!' o 'pip ') se comentan para que el
    script se pueda ejecutar con Python.

    Args:
        name (str): Nombre del notebook, con o sin la extensión .ipynb.

    Returns:
        str: Ruta del archivo .py generado.
    """
    name = name[:-6] if name.endswith(".ipynb") else name
    path = f"{name}.ipynb"
    if not os.path.isfile(path):
        # Permite nombres con sufijos de descarga, p. ej. "Entrega 1 (4).ipynb"
        candidates = sorted(glob(f"{glob_escape(name)}*.ipynb"))
        if not candidates:
            raise FileNotFoundError(f"No se encontró el notebook '{path}'")
        path = candidates[0]

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    blocks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!", "pip ")) or "converter(" in stripped:
                line = "# " + line
            lines.append(line)
        blocks.append("\n".join(lines))

    out_path = f"{name}.py"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(blocks) + "\n")
    print(f"Notebook '{path}' convertido a '{out_path}'")
    return out_path


def glob_escape(text: str) -> str:
    """Escapa los caracteres especiales de glob ('[', ']', '*', '?')."""
    return "".join(f"[{c}]" if c in "[]*?" else c for c in text)
