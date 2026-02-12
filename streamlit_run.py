"""Entry point for Streamlit so the app package resolves correctly."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Delegate to the real app
from app.app import main

if __name__ == "__main__":
    main()
