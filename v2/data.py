"""Bounded input handling and deterministic demo with a real Simpson reversal."""
from io import BytesIO
import hashlib
from pathlib import Path
import csv
from itertools import islice
import zipfile
import numpy as np
import pandas as pd

MAX_ROWS = 20000
MAX_COLS = 80


def validate(df):
    if df.empty or len(df) < 12 or len(df.columns) < 2:
        raise ValueError("Use a dataset with at least 12 rows and two columns.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLS:
        raise ValueError(f"Demo limits: {MAX_ROWS:,} rows and {MAX_COLS} columns.")
    names = [str(c).strip() for c in df.columns]
    if len(names) != len(set(names)) or any(not c or len(c) > 128 for c in names):
        raise ValueError("Column names must be unique, non-empty, and at most 128 characters.")
    df = df.copy()
    df.columns = names
    return df.replace([np.inf, -np.inf], np.nan)


def load_bytes(raw, name):
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("Upload a file smaller than 10 MB.")
    if name.lower().endswith('.csv'):
        # Reject duplicate headers before pandas silently mangles them.
        try:
            header = next(csv.reader(raw.decode('utf-8-sig').splitlines()))
        except (UnicodeDecodeError, StopIteration) as exc:
            raise ValueError('Use a non-empty UTF-8 CSV file.') from exc
        if len(header) > MAX_COLS:
            raise ValueError(f'Use at most {MAX_COLS} columns.')
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names.")
        df = pd.read_csv(BytesIO(raw), nrows=MAX_ROWS + 1)
    elif name.lower().endswith('.xlsx'):
        try:
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                if len(archive.infolist()) > 1000 or sum(i.file_size for i in archive.infolist()) > 50 * 1024 * 1024:
                    raise ValueError('This workbook expands beyond the supported size. Export a small CSV instead.')
            from openpyxl import load_workbook
            book = load_workbook(BytesIO(raw), read_only=True, data_only=True, keep_links=False)
            try:
                sheet = book.worksheets[0]
                if sheet.max_column and sheet.max_column > MAX_COLS:
                    raise ValueError(f'Use at most {MAX_COLS} columns.')
                rows = sheet.iter_rows(values_only=True, max_col=MAX_COLS+1)
                header = list(next(rows))
                while header and header[-1] is None:
                    header.pop()
                if not header or len(header) > MAX_COLS or any(c is None for c in header):
                    raise ValueError('The first worksheet needs a single, complete header row.')
                if len({str(c).strip() for c in header}) != len(header):
                    raise ValueError('Excel contains duplicate column names.')
                values = [tuple(r[:len(header)]) for r in islice(rows, MAX_ROWS+1)]
                df = pd.DataFrame(values, columns=header)
            finally:
                book.close()
        except (zipfile.BadZipFile, KeyError, StopIteration, IndexError) as exc:
            raise ValueError('The Excel file is empty or damaged. Try exporting it as CSV.') from exc
    else:
        raise ValueError("Supported formats: CSV and XLSX.")
    return validate(df)


def fingerprint(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


def engine_fingerprint():
    root=Path(__file__).resolve().parent
    paths=sorted(root.glob('*.py'))+[root.parent/'app/modules/feature_engineering/feature_service.py']
    digest=hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode());digest.update(path.read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def demo_data():
    rng = np.random.default_rng(42)
    rows = []
    # Within BOTH segments, premium support has lower churn. Aggregated it
    # looks worse because most premium customers belong to the high-risk cohort.
    for segment, premium, n, churns in [
        ('New customers', 'Premium', 240, 144),
        ('New customers', 'Standard', 60, 42),
        ('Established', 'Premium', 60, 6),
        ('Established', 'Standard', 240, 48),
    ]:
        outcomes = np.array([1] * churns + [0] * (n-churns))
        rng.shuffle(outcomes)
        for churn in outcomes:
            rows.append({'customer_id':f'C{len(rows)+1:04}',
                         'support_plan':premium, 'customer_segment':segment,
                         'tenure_months':int(rng.integers(1,7) if segment=='New customers' else rng.integers(18,73)),
                         'monthly_spend':round(float(rng.normal(80 if premium=='Premium' else 55,12)),2),
                         'churn':int(churn), 'closed_account_flag':int(churn)})
    df = pd.DataFrame(rows).sample(frac=1,random_state=42).reset_index(drop=True)
    df.loc[[7,34,92,140,255,399], 'monthly_spend'] = np.nan
    return df
