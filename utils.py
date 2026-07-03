import pandas as pd


def load_interns(file_path):
    """Load intern dataset."""
    return pd.read_csv(file_path)


def load_jobs(file_path):
    """Load job dataset."""
    return pd.read_csv(file_path)