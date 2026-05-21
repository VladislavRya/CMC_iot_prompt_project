"""Shared data loading for Kaggle CTU-IoT / UNSW-NB15 style CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Primary dataset for this project (CTU-IoT Malware Capture, Zeek conn.log)
DEFAULT_CTU_CSV = "CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv"

LABEL_COL = "label"

# IDs, timestamps, IPs, ports, auxiliary label columns
DROP_COLS = {
    "id",
    "attack_cat",
    "srcip",
    "dstip",
    "sport",
    "dsport",
    "stime",
    "ltime",
    # CTU / Zeek conn.log
    "ts",
    "uid",
    "id_orig_h",
    "id_orig_p",
    "id_resp_h",
    "id_resp_p",
    "detailed_label",
    "tunnel_parents",
    "history",
}

LABEL_MAP = {
    "malicious": 1,
    "benign": 0,
    "attack": 1,
    "normal": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
}

# CTU conn.log numeric fields (UNSW names included for synthetic data)
NUMERIC_HINTS = {
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
}


def _normalize_columns(columns: pd.Index) -> list[str]:
    return [
        str(c).strip().lower().replace(".", "_").replace("-", "_")
        for c in columns
    ]


def _detect_separator(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    return "|" if header.count("|") > header.count(",") else ","


def find_csv(data_dir: Path, filename: str | None = None) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Download Kaggle dataset or run: python scripts/generate_synthetic_data.py"
        )

    if filename:
        path = data_dir / filename
        if path.exists():
            return path
        raise FileNotFoundError(f"Requested CSV not found: {path}")

    preferred = [
        DEFAULT_CTU_CSV,
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
        "train.csv",
        "network_malware.csv",
    ]
    for name in preferred:
        path = data_dir / name
        if path.exists():
            return path

    # Prefer CTU captures over synthetic demo file
    ctu_files = sorted(data_dir.glob("CTU-IoT-Malware-Capture-*conn.log.labeled.csv"))
    if ctu_files:
        return ctu_files[0]

    csv_files = sorted(
        p for p in data_dir.glob("*.csv") if p.name != "synthetic_unsw_nb15.csv"
    )
    if not csv_files:
        synthetic = data_dir / "synthetic_unsw_nb15.csv"
        if synthetic.exists():
            return synthetic
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    return csv_files[0]


def _coerce_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if col in NUMERIC_HINTS or out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().mean() >= 0.05:
                out[col] = converted
    return out


def _drop_uninformative_columns(X: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in X.columns:
        if X[col].notna().sum() == 0:
            continue
        if X[col].nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    return X[keep]


def _encode_labels(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(LABEL_MAP)
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique())[:10]
        raise ValueError(f"Unknown label values: {unknown}")
    return mapped.astype(int)


def _replace_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Zeek exports use '-' for empty fields."""
    out = df.replace("-", np.nan)
    out = out.replace("", np.nan)
    return out


def load_dataset(
    data_dir: Path,
    filename: str | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    path = find_csv(data_dir, filename=filename)
    sep = _detect_separator(path)

    if max_rows is not None:
        df = pd.read_csv(path, sep=sep, nrows=max_rows, low_memory=False)
    else:
        df = pd.read_csv(path, sep=sep, low_memory=False)

    df.columns = _normalize_columns(df.columns)
    df = _replace_placeholders(df)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Column '{LABEL_COL}' not found in {path.name}. Got: {list(df.columns)}")

    y = _encode_labels(df[LABEL_COL])

    if max_rows is not None and len(df) >= max_rows:
        # If we only read head, skip; otherwise subsample fairly below
        pass
    elif max_rows is not None and len(df) > max_rows:
        idx = (
            pd.concat([y], axis=1)
            .groupby(LABEL_COL, group_keys=False)
            .apply(lambda g: g.sample(frac=max_rows / len(g), random_state=random_state))
            .index
        )
        df = df.loc[idx]
        y = y.loc[idx]

    feature_cols = [c for c in df.columns if c not in DROP_COLS and c != LABEL_COL]
    X = df[feature_cols].copy()
    X = _coerce_numeric_features(X)
    X = _drop_uninformative_columns(X)
    return X, y


def dataset_info(data_dir: Path, filename: str | None = None) -> dict:
    path = find_csv(data_dir, filename=filename)
    sep = _detect_separator(path)
    sample = pd.read_csv(path, sep=sep, nrows=5000, low_memory=False)
    sample.columns = _normalize_columns(sample.columns)
    return {
        "path": str(path),
        "name": path.name,
        "separator": repr(sep),
        "columns": list(sample.columns),
        "sample_rows": len(sample),
    }


def generate_synthetic(
    n_samples: int = 8000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic UNSW-NB15-like data for demo without Kaggle download."""
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * 0.35)

    def block(n: int, attack: bool) -> pd.DataFrame:
        proto = rng.choice(["tcp", "udp", "icmp"], n, p=[0.7, 0.25, 0.05])
        service = rng.choice(["http", "dns", "ssh", "-", "ftp"], n)
        state = rng.choice(["FIN", "CON", "INT", "REQ"], n)
        base = {
            "dur": rng.exponential(0.5 if attack else 1.2, n),
            "spkts": rng.integers(1, 50, n),
            "dpkts": rng.integers(1, 50, n),
            "sbytes": rng.integers(100, 50000, n),
            "dbytes": rng.integers(100, 50000, n),
            "proto": proto,
            "service": service,
            "state": state,
        }
        numeric_extra = [
            "rate", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
            "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
            "synack", "ackdat", "smean", "dmean", "trans_depth",
            "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
            "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
            "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
        ]
        for col in numeric_extra:
            if col in base:
                continue
            scale = 3.0 if attack else 1.0
            base[col] = rng.exponential(scale, n)
        for col in ("is_ftp_login", "is_sm_ips_ports"):
            base[col] = rng.integers(0, 2, n)
        if attack:
            base["sload"] = base.get("sload", rng.exponential(3, n)) * 2
            base["rate"] = base.get("rate", rng.exponential(3, n)) * 1.8
        return pd.DataFrame(base)

    X_normal = block(n_samples - n_attack, attack=False)
    X_attack = block(n_attack, attack=True)
    X = pd.concat([X_normal, X_attack], ignore_index=True)
    y = pd.Series([0] * (n_samples - n_attack) + [1] * n_attack, name=LABEL_COL)
    perm = rng.permutation(len(y))
    return X.iloc[perm].reset_index(drop=True), y.iloc[perm].reset_index(drop=True)
