import json
from pathlib import Path

NOTEBOOK = Path("notebooks/01_roboops_workshop.ipynb")


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_code_cells_compile() -> None:
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{NOTEBOOK}:code-cell-{index}", "exec")


def test_notebook_has_no_saved_error_outputs() -> None:
    notebook = load_notebook()
    errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors
