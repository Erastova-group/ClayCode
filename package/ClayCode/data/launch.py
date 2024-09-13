import shutil
import subprocess
from pathlib import Path

import nbformat
from ClayCode import PathOrStr
from ClayCode.core.utils import execute_shell_command

uc_writer_script = Path(__file__).parent / "uc_maker.ipynb"

shutil.copy(uc_writer_script, "ucwscr.ipynb")
# app = JupyterNotebookApp()
# nb = nbformat.read("ucwscr.ipynb", nbformat.NO_CONVERT)
# nb['cells'].append(nbformat.v4.new_code_cell('from ClayCode import ClayCodeLogger'))
# nbformat.write(nb, 'ucwscr.ipynb')
p = subprocess.Popen(["jupyter", "notebook", "ucwscr.ipynb"])
# p = execute_shell_command(["jupyter", "notebook", "ucwscr.ipynb"])
# nb['cells'].append(nbformat.v4.new_code_cell('from ClayCode import ClayCodeLogger'))
# nbformat.write(nb, 'ucwscr.ipynb')
print("done")


class Notebookwriter:
    def __init__(self, outpath: PathOrStr, name: str):
        self.outpath = Path(outpath)
        self.name = name
        self.nb = nbformat.read(uc_writer_script, nbformat.NO_CONVERT)
        self.nb["cells"].append(
            nbformat.v4.new_code_cell(
                f'from ClayCode import ClayCodeLogger\nlogger = ClayCodeLogger("{self.name}")'
            )
        )
        self.nb["cells"].append(
            nbformat.v4.new_code_cell(f'logger.finfo("Writing notebook")')
        )
        self.nb["cells"].append(
            nbformat.v4.new_code_cell(
                f'logger.finfo("Finished writing notebook")'
            )
        )
        # nbformat.write(self.nb, self.outpath / f'{self.name}.ipynb')
        # self.nb = nbformat.read(self.outpath / f'{self.name}.ipynb', nbformat.NO_CONVERT)

    def write(self):
        nbformat.write(self.nb, self.outpath / f"{self.name}.ipynb")

    def open(self):
        execute_shell_command(
            f'jupyter notebook {self.outpath / f"{self.name}.ipynb &"}'
        )
