import re
from pathlib import Path
import pandas as pd

def load_polar(file_path: str) -> pd.DataFrame:
    """
    Read a single XFOIL polar text file and return a DataFrame with columns ['Re', 'Ncrit', 'CL', 'CD'].
    """
    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract Re and Ncrit from the header
    Re = None
    Ncrit = None
    for line in lines:
        if 'Re =' in line and 'Ncrit' in line:
            # Parse Reynolds number in format like 'Re =  0.050 e 6'
            re_match = re.search(r'Re\s*=\s*([0-9\.]+)\s*e\s*(\d+)', line)
            if re_match:
                mantissa = float(re_match.group(1))
                exponent = int(re_match.group(2))
                Re = mantissa * 10**exponent
            else:
                # Fallback if no exponent
                re_match = re.search(r'Re\s*=\s*([0-9\.]+)', line)
                Re = float(re_match.group(1)) if re_match else None

            ncrit_match = re.search(r'Ncrit\s*=\s*([0-9\.]+)', line)
            Ncrit = float(ncrit_match.group(1)) if ncrit_match else None
            break

    if Re is None or Ncrit is None:
        raise ValueError(f"Could not parse Re/Ncrit from file: {file_path}")

    # Locate the table header line (starts with 'alpha')
    header_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith('alpha'):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find data header in file: {file_path}")

    # Skip the header line and the underline, then read the numeric data
    skiprows = header_idx + 2
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=skiprows,
        names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr']
    )

    # Keep only the columns of interest and add Re, Ncrit
    df = df[['alpha', 'CL', 'CD']].copy()
    df['Re'] = Re
    df['Ncrit'] = Ncrit

    return df


def load_all_polars(dir_path: str) -> pd.DataFrame:
    """
    Read all .txt polar files in the given directory and concatenate into a single DataFrame.
    """
    p = Path(dir_path)
    if not p.is_dir():
        raise ValueError(f"Provided path is not a directory: {dir_path}")

    all_dfs = []
    for file in p.glob('*.txt'):
        try:
            df = load_polar(str(file))
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: skipping {file.name}: {e}")

    if not all_dfs:
        raise ValueError(f"No valid polar files found in directory: {dir_path}")

    return pd.concat(all_dfs, ignore_index=True)


