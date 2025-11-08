from pathlib import Path
import re
import io
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve()
DATA_DIR = HERE.parent
IN_DIR = DATA_DIR / "datasetTemp"        # .per input
OUT_DIR = DATA_DIR / "tempPerCountry"    # CSV output
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_MAP = {m:i+1 for i,m in enumerate(MONTHS)}
MISSING = -999.0

def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip())
    return s.strip("_") or "UNKNOWN"

def extract_country_from_header(lines: list[str]) -> str | None:
    for ln in lines[:8]:
        m = re.search(r"Country\s*=\s*([^:]+)", ln)
        if m:
            return m.group(1).strip()
    return None

def extract_country_from_filename(path: Path) -> str:
    m = re.search(r"\.(?P<name>[^.]+)\.tmp\.per$", path.name, flags=re.IGNORECASE)
    if m:
        return m.group("name")
    parts = path.stem.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return path.stem

def parse_per_file(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()

    country = extract_country_from_header(lines) or extract_country_from_filename(path)

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("YEAR"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Header line with 'YEAR' not found in {path.name}")

    table_str = "\n".join(lines[header_idx:])
    df = pd.read_fwf(io.StringIO(table_str))

    missing_months = [m for m in MONTHS if m not in df.columns]
    if missing_months:
        raise ValueError(f"{path.name}: missing month columns: {missing_months}")

    df = df[["YEAR"] + MONTHS].copy()

    df.replace(MISSING, np.nan, inplace=True)

    long = df.melt(id_vars=["YEAR"], value_vars=MONTHS, var_name="month_str", value_name="temp_c")
    long["month"] = long["month_str"].map(MONTH_MAP).astype(int)
    long["year"] = long["YEAR"].astype(int)
    long["date"] = pd.to_datetime(dict(year=long["year"], month=long["month"], day=15), errors="coerce")
    long["country"] = country

    long = long[["date","year","month","temp_c","country"]].sort_values(["date"])
    return long

def main():
    per_files = sorted(IN_DIR.glob("*.per"))
    if not per_files:
        print(f"[ERROR] No .per files in {IN_DIR}")
        return

    total_rows = 0
    written = 0
    for fp in per_files:
        try:
            df = parse_per_file(fp)
        except Exception as e:
            print(f"[WARN] skip {fp.name}: {e}")
            continue

        country = df["country"].iloc[0]
        out = OUT_DIR / f"{safe_name(country)}.csv"
        df.to_csv(out, index=False)
        written += 1
        total_rows += len(df)
        if written % 10 == 0:
            print(f"[OK] {written} files written... (last: {out.name})")

    print(f"[DONE] files: {written} | total rows: {total_rows} → {OUT_DIR}")

if __name__ == "__main__":
    main()
