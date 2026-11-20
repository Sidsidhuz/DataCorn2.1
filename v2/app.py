"""Streamlit entrypoint. Run python -m v2.launch --mode local."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2.ui.shell import render

render()
