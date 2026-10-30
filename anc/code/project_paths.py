from pathlib import Path
import os
MODULE_DIR = Path(__file__).resolve().parent
DATA_CORE = Path(os.environ.get('REGRET_CORE_DATA', MODULE_DIR / 'data'))
DATA_EXTRA = Path(os.environ.get('REGRET_EXTRA_DATA', MODULE_DIR / 'data'))
