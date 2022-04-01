import os
import sys
module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

print(os.path.abspath(os.path.join('..')))

from src import cloppa