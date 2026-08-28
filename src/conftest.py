import sys
from pathlib import Path
 
# Add the repo's src/ directory to sys.path so tests can import modules
# like `preprocess_data` and `model_training` directly.
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))