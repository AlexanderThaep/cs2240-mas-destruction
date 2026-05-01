import os
import sys
import subprocess
from pathlib import Path

pip_cmd = "pip"
requirements_path = Path("requirements.txt")

if not requirements_path.is_file():
    print(f"Error: Requirements not found.")
    sys.exit(1)

result = subprocess.run(
    [str(pip_cmd), "install", "-r", str(os.path.abspath(requirements_path))],
    cwd=str(os.getcwd()),
    stdout=sys.stdout,
    stderr=sys.stderr
)