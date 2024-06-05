import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
sys.path.append('../')
from .hook import CLAP_Module