#!/usr/bin/env python3
"""Run the SafeMolGen-DrugOracle Streamlit application."""

import os
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    os.system("streamlit run app/app.py")


if __name__ == "__main__":
    main()
