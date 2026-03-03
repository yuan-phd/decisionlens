"""
setup_data.py — DecisionLENS data setup script.

Data source priority (first available wins):
  1. data/processed/  — pre-existing parquet files from a previous run
  2. data/deployment/ — parquet sample created by create_deployment_sample.py
  3. HuggingFace Hub  — yuanphd/decisionlens-aact-sample (auto-downloaded)
  4. Local zip        — AACT flat-file zip placed in data/raw/
  5. Synthetic        — seeded 10 K-study fallback for demo / CI

Usage:
    python setup_data.py [--data-dir ./data] [--force-synthetic]

To use a local AACT zip instead of HuggingFace:
    Download from https://aact.ctti-clinicaltrials.org/downloads
    Place the zip (e.g. 20260228_clinical_trials.zip) in data/raw/
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Tables we actually need — each becomes one parquet file.
REQUIRED_TABLES = [
    "studies",
    "calculated_values",
    "eligibilities",
    "designs",
    "facilities",
    "countries",
    "sponsors",
    "conditions",
    "interventions",
    "outcome_counts",
]

SYNTHETIC_N_STUDIES = 10_000
SYNTHETIC_SEED = 42

#: HuggingFace dataset repo containing the pre-built AACT parquet sample.
HF_REPO: str = "yuanphd/decisionlens-aact-sample"


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------


def create_directories(data_dir: Path) -> None:
    """Create the required project directories if they don't exist."""
    for subdir in ("raw", "processed"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    log.info("Directories ready: %s", data_dir)


# ---------------------------------------------------------------------------
# Helpers: detect and load already-prepared parquet directories
# ---------------------------------------------------------------------------


def _dir_has_data(directory: Path) -> bool:
    """Return True if *directory* contains a non-empty studies.parquet."""
    p = directory / "studies.parquet"
    return p.exists() and p.stat().st_size > 0


def _load_parquet_tables(src_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all REQUIRED_TABLES parquet files from *src_dir* that exist."""
    tables: dict[str, pd.DataFrame] = {}
    for table in REQUIRED_TABLES:
        p = src_dir / f"{table}.parquet"
        if not p.exists():
            log.warning("  %-22s  not found in %s — skipping", table, src_dir)
            continue
        df = pd.read_parquet(p)
        tables[table] = df
        log.info("  loaded  %-22s  %7s rows", table, f"{len(df):,}")
    return tables


def _copy_to_processed(
    src_dir: Path, dest_dir: Path
) -> dict[str, pd.DataFrame]:
    """
    Copy parquet files from *src_dir* into *dest_dir*.

    Skips tables already present in *dest_dir* to avoid redundant I/O.
    Returns the loaded tables dict (read from dest after copying).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for table in REQUIRED_TABLES:
        src = src_dir / f"{table}.parquet"
        dst = dest_dir / f"{table}.parquet"
        if not src.exists():
            continue
        if dst.exists():
            log.info("  %-22s  already in processed/ — skipping copy", table)
            continue
        dst.write_bytes(src.read_bytes())
        log.info("  copied  %-22s  (%.1f MB)", table, dst.stat().st_size / 1e6)
    return _load_parquet_tables(dest_dir)


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------


def download_from_huggingface(
    processed_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Download the AACT parquet sample from HuggingFace Hub.

    Files are downloaded directly into *processed_dir* (skipping any table
    whose parquet already exists there).  Returns an empty dict if
    ``huggingface_hub`` is not installed or the download fails.

    Args:
        processed_dir: Destination directory for the parquet files.

    Returns:
        Dict of {table_name: DataFrame}, or {} on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning(
            "huggingface_hub is not installed — skipping HuggingFace download. "
            "Install with: pip install huggingface_hub"
        )
        return {}

    processed_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading from HuggingFace: %s", HF_REPO)

    failed: list[str] = []
    for table in REQUIRED_TABLES:
        filename = f"{table}.parquet"
        dest = processed_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("  %-22s  already exists — skipping download", filename)
            continue
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(processed_dir),
            )
            log.info("  downloaded  %s  (%.1f MB)", filename, dest.stat().st_size / 1e6)
        except Exception as exc:
            log.error("  failed to download %s: %s", filename, exc)
            failed.append(table)

    if failed:
        log.error("HuggingFace download incomplete — missing: %s", ", ".join(failed))
        return {}

    return _load_parquet_tables(processed_dir)


# ---------------------------------------------------------------------------
# Local zip extraction
# ---------------------------------------------------------------------------


def find_local_zip(raw_dir: Path) -> Path | None:
    """
    Find any .zip file in *raw_dir*.

    If multiple zips are present, the most recently modified one is used.

    Args:
        raw_dir: Directory to search (typically data/raw/).

    Returns:
        Path to the zip file, or None if not found.
    """
    candidates = sorted(
        raw_dir.glob("*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        log.info("Found zip: %s", candidates[0].name)
        if len(candidates) > 1:
            log.info("  (%d other zip(s) in directory — using most recent)", len(candidates) - 1)
    return candidates[0] if candidates else None


def extract_local_zip(raw_dir: Path) -> bool:
    """
    Extract AACT pipe-delimited .txt files from a local zip into *raw_dir*.

    Looks for a zip matching *clinical_trials*.zip in *raw_dir*.  Each .txt
    file in the zip that matches a required table name is written out as
    ``<table>.txt``.

    Args:
        raw_dir: Directory containing the zip and where .txt files are written.

    Returns:
        True if at least one required table was extracted successfully.
    """
    zip_path = find_local_zip(raw_dir)
    if zip_path is None:
        log.warning(
            "No .zip file found in %s. "
            "Download from https://aact.ctti-clinicaltrials.org/downloads "
            "and place it in that directory.",
            raw_dir,
        )
        return False

    log.info("Inspecting %s …", zip_path.name)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            txt_names = [n for n in all_names if n.lower().endswith(".txt")]
            log.info("  Zip contains %d total entries, %d .txt files", len(all_names), len(txt_names))
            print("\n  First 5 entries in zip:")
            for name in all_names[:5]:
                info = zf.getinfo(name)
                print(f"    {name:<45}  {info.file_size / 1e6:6.1f} MB")
            print()
    except zipfile.BadZipFile as exc:
        log.error("Cannot open zip file: %s", exc)
        return False

    log.info("Extracting tables from %s …", zip_path.name)
    extracted = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Build a lowercase stem → archive name map for all .txt entries
            txt_entries: dict[str, str] = {
                Path(name).stem.lower(): name
                for name in zf.namelist()
                if name.lower().endswith(".txt")
            }

            for table in REQUIRED_TABLES:
                archive_name = txt_entries.get(table)
                if archive_name is None:
                    log.warning("  Table not found in zip: %s", table)
                    continue
                dest = raw_dir / f"{table}.txt"
                dest.write_bytes(zf.read(archive_name))
                size_mb = dest.stat().st_size / 1e6
                log.info("  Extracted %-25s  %.1f MB", table, size_mb)
                extracted += 1

    except zipfile.BadZipFile as exc:
        log.error("Cannot open zip file: %s", exc)
        return False

    if extracted == 0:
        log.error("No required tables were found in the zip.")
        return False

    log.info("Extraction complete: %d/%d tables extracted.", extracted, len(REQUIRED_TABLES))
    return True


# ---------------------------------------------------------------------------
# Phase / status normalisation (applied to studies before writing parquet)
# ---------------------------------------------------------------------------

_PHASE_REPAIR_MAP: dict[str, str] = {
    # No-space / all-uppercase variants (exact AACT raw strings)
    "PHASE1":            "Phase 1",
    "Phase1":            "Phase 1",
    "PHASE 1":           "Phase 1",
    "EARLY_PHASE1":      "Phase 1",
    "Early_Phase1":      "Phase 1",
    "Early Phase 1":     "Phase 1",
    # Phase 2
    "PHASE2":            "Phase 2",
    "Phase2":            "Phase 2",
    "PHASE 2":           "Phase 2",
    # Phase 3
    "PHASE3":            "Phase 3",
    "Phase3":            "Phase 3",
    "PHASE 3":           "Phase 3",
    # Phase 4
    "PHASE4":            "Phase 4",
    "Phase4":            "Phase 4",
    "PHASE 4":           "Phase 4",
    # Combined phases
    "PHASE1/PHASE2":     "Phase 1/Phase 2",
    "Phase1/Phase2":     "Phase 1/Phase 2",
    "PHASE 1/PHASE 2":   "Phase 1/Phase 2",
    "Phase 1 / Phase 2": "Phase 1/Phase 2",
    "Phase 1/2":         "Phase 1/Phase 2",
    "PHASE2/PHASE3":     "Phase 2/Phase 3",
    "Phase2/Phase3":     "Phase 2/Phase 3",
    "PHASE 2/PHASE 3":   "Phase 2/Phase 3",
    "Phase 2 / Phase 3": "Phase 2/Phase 3",
    "Phase 2/3":         "Phase 2/Phase 3",
    # Null-like strings
    "None":              "N/A",
    "nan":               "N/A",
    "":                  "N/A",
}


def _normalise_studies_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise phase and overall_status values in the raw studies table.

    Applied before writing to parquet so the stored files are already clean.
    Mirrors the repair logic in ``src/data_pipeline.py::engineer_features``.
    """
    if "phase" in df.columns:
        df["phase"] = (
            df["phase"]
            .astype(str)
            .str.strip()
            .replace(_PHASE_REPAIR_MAP)
            .fillna("N/A")
        )
    if "overall_status" in df.columns:
        df["overall_status"] = df["overall_status"].astype(str).str.strip().str.title()
    return df


# ---------------------------------------------------------------------------
# Load raw TXT → parquet
# ---------------------------------------------------------------------------


def load_raw_tables(raw_dir: Path, processed_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Read pipe-delimited AACT txt files and save as parquet.

    Returns a dict of {table_name: DataFrame}.
    """
    tables: dict[str, pd.DataFrame] = {}
    for table in REQUIRED_TABLES:
        txt_path = raw_dir / f"{table}.txt"
        if not txt_path.exists():
            log.warning("Missing raw file: %s (skipping)", txt_path)
            continue
        log.info("Loading %s …", txt_path.name)
        try:
            df = pd.read_csv(txt_path, sep="|", low_memory=False)
            if table == "studies":
                df = _normalise_studies_df(df)
            tables[table] = df
            pq_path = processed_dir / f"{table}.parquet"
            df.to_parquet(pq_path, index=False)
            log.info("  %d rows → %s", len(df), pq_path.name)
        except Exception as exc:
            log.error("Failed to load %s: %s", table, exc)
    return tables


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def _random_nct_ids(n: int, rng: np.random.Generator) -> list[str]:
    nums = rng.integers(10_000_000, 99_999_999, size=n)
    return [f"NCT{x:08d}" for x in nums]


def generate_synthetic_data(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Generate a realistic synthetic AACT-like dataset for demo purposes.

    Distributions are calibrated to approximate real AACT marginals.
    Returns a dict of {table_name: DataFrame}.
    """
    log.info("Generating synthetic dataset (%d studies) …", SYNTHETIC_N_STUDIES)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n = SYNTHETIC_N_STUDIES
    nct_ids = _random_nct_ids(n, rng)

    # ---- studies -----------------------------------------------------------
    phases = rng.choice(
        ["Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4", "N/A"],
        size=n,
        p=[0.20, 0.05, 0.28, 0.05, 0.25, 0.10, 0.07],
    )
    statuses = rng.choice(
        ["Completed", "Terminated", "Active, not recruiting", "Recruiting", "Withdrawn"],
        size=n,
        p=[0.55, 0.15, 0.10, 0.15, 0.05],
    )
    start_year = rng.integers(2008, 2024, size=n)
    start_month = rng.integers(1, 13, size=n)
    start_dates = pd.to_datetime(
        {"year": start_year, "month": start_month, "day": 1}
    )
    duration_days = rng.integers(180, 2000, size=n)
    completion_dates = start_dates + pd.to_timedelta(duration_days, unit="D")

    target_enrollment = rng.integers(20, 2000, size=n).astype(float)
    # Actual enrollment: mostly close to target with some over/underperformers
    _ratio = np.clip(rng.normal(loc=0.92, scale=0.25, size=n), 0.05, 2.0)
    actual_enrollment = (target_enrollment * _ratio).round().astype(float)

    # enrollment      = planned target (enrollment_type = 'Anticipated')
    # actual_enrollment = what was actually enrolled — only known for completed trials.
    # Terminated trials also have a partial actual; others (recruiting, etc.) → NaN.
    _has_actual = np.isin(statuses, ["Completed", "Terminated"])

    studies = pd.DataFrame({
        "nct_id": nct_ids,
        "study_type": rng.choice(["Interventional"], size=n),
        "overall_status": statuses,
        "phase": phases,
        "enrollment": target_enrollment,          # always the planned target
        "enrollment_type": "Anticipated",         # enrollment col = anticipated target
        "actual_enrollment": np.where(            # actual only when outcome is known
            _has_actual, actual_enrollment, np.nan
        ),
        "brief_title": [f"Synthetic Study {i}" for i in range(n)],
        "start_date": start_dates,
        "completion_date": completion_dates,
        "has_dmc": rng.choice(["t", "f"], size=n, p=[0.6, 0.4]),
        "is_fda_regulated_drug": rng.choice(["t", "f"], size=n, p=[0.7, 0.3]),
        "is_fda_regulated_device": rng.choice(["t", "f"], size=n, p=[0.2, 0.8]),
        "why_stopped": np.where(statuses == "Terminated",
                                rng.choice(["Safety", "Lack of efficacy", "Business decision"], size=n),
                                None),
    })

    # ---- calculated_values -------------------------------------------------
    n_facilities = rng.integers(1, 300, size=n)
    calculated_values = pd.DataFrame({
        "nct_id": nct_ids,
        "actual_duration": duration_days,
        "number_of_facilities": n_facilities,
        "number_of_sae_subjects": rng.integers(0, 50, size=n),
        "minimum_age_num": rng.integers(18, 65, size=n).astype(float),
        "maximum_age_num": rng.integers(65, 90, size=n).astype(float),
    })

    # ---- eligibilities -----------------------------------------------------
    n_criteria = rng.integers(3, 40, size=n)
    eligibilities = pd.DataFrame({
        "nct_id": nct_ids,
        "gender": rng.choice(["All", "Female", "Male"], size=n, p=[0.75, 0.15, 0.10]),
        "minimum_age": [f"{a} Years" for a in rng.integers(18, 55, size=n)],
        "maximum_age": [f"{a} Years" for a in rng.integers(60, 90, size=n)],
        "healthy_volunteers": rng.choice(["No", "Yes"], size=n, p=[0.85, 0.15]),
        "criteria": [
            f"Inclusion Criteria:\n- Criterion 1\n- Criterion 2\n"
            f"Exclusion Criteria:\n" + "\n".join([f"- Excl {j}" for j in range(k)])
            for k in n_criteria
        ],
    })

    # ---- designs -----------------------------------------------------------
    designs = pd.DataFrame({
        "nct_id": nct_ids,
        "intervention_model": rng.choice(
            ["Parallel Assignment", "Crossover Assignment", "Single Group Assignment", "Factorial Assignment"],
            size=n, p=[0.60, 0.15, 0.20, 0.05],
        ),
        "primary_purpose": rng.choice(
            ["Treatment", "Prevention", "Diagnostic", "Supportive Care", "Basic Science"],
            size=n, p=[0.65, 0.15, 0.08, 0.07, 0.05],
        ),
        "masking": rng.choice(
            ["None (Open Label)", "Single", "Double", "Triple", "Quadruple"],
            size=n, p=[0.35, 0.15, 0.30, 0.10, 0.10],
        ),
        "allocation": rng.choice(["Randomized", "Non-Randomized", "N/A"], size=n, p=[0.75, 0.15, 0.10]),
    })

    # ---- facilities (one-to-many) ------------------------------------------
    country_pool = [
        "United States", "Germany", "United Kingdom", "France", "Canada",
        "Japan", "China", "Australia", "Spain", "Italy", "Brazil", "India",
        "Netherlands", "Belgium", "South Korea",
    ]
    facility_rows = []
    for nct_id, n_fac in zip(nct_ids, n_facilities):
        n_fac_clipped = min(int(n_fac), 20)  # cap per study for size
        countries_chosen = rng.choice(country_pool, size=n_fac_clipped, replace=True)
        for country in countries_chosen:
            facility_rows.append({"nct_id": nct_id, "country": country,
                                   "status": rng.choice(["Individual Site", "Withdrawn"], p=[0.9, 0.1])})
    facilities = pd.DataFrame(facility_rows)

    # ---- countries (unique per study) --------------------------------------
    countries_rows = (
        facilities[["nct_id", "country"]]
        .drop_duplicates()
        .rename(columns={"country": "name"})
        .assign(removed="f")
    )

    # ---- sponsors ----------------------------------------------------------
    sponsor_names = [f"Sponsor_{rng.integers(1, 500)}" for _ in range(n)]
    sponsors = pd.DataFrame({
        "nct_id": nct_ids,
        "name": sponsor_names,
        "agency_class": rng.choice(["Industry", "NIH", "U.S. Fed", "Other"], size=n, p=[0.55, 0.15, 0.05, 0.25]),
        "lead_or_collaborator": rng.choice(["lead", "collaborator"], size=n, p=[0.8, 0.2]),
    })

    # ---- conditions --------------------------------------------------------
    condition_pool = [
        "Type 2 Diabetes", "Non-Small Cell Lung Cancer", "Breast Cancer",
        "Alzheimer Disease", "Chronic Obstructive Pulmonary Disease",
        "Heart Failure", "Major Depressive Disorder", "Rheumatoid Arthritis",
        "Multiple Sclerosis", "Colorectal Cancer", "Hypertension",
        "Atrial Fibrillation", "Parkinson Disease", "Asthma", "Psoriasis",
        "Crohn Disease", "Ulcerative Colitis", "Ovarian Cancer", "Leukemia",
        "HIV Infections",
    ]
    conditions = pd.DataFrame({
        "nct_id": nct_ids,
        "name": rng.choice(condition_pool, size=n),
        "downcase_name": [c.lower() for c in rng.choice(condition_pool, size=n)],
    })

    # ---- interventions -----------------------------------------------------
    intervention_types = rng.choice(
        ["Drug", "Biological", "Device", "Procedure", "Behavioral", "Other"],
        size=n, p=[0.52, 0.18, 0.10, 0.08, 0.07, 0.05],
    )
    interventions = pd.DataFrame({
        "nct_id": nct_ids,
        "intervention_type": intervention_types,
        "name": [f"Intervention_{rng.integers(1, 200)}" for _ in range(n)],
    })

    # ---- outcome_counts ----------------------------------------------------
    outcome_counts = pd.DataFrame({
        "nct_id": rng.choice(nct_ids, size=n * 2, replace=True),
        "outcome_type": rng.choice(["Primary", "Secondary"], size=n * 2),
        "count": rng.integers(1, 10, size=n * 2),
    })

    tables = {
        "studies": studies,
        "calculated_values": calculated_values,
        "eligibilities": eligibilities,
        "designs": designs,
        "facilities": facilities,
        "countries": countries_rows,
        "sponsors": sponsors,
        "conditions": conditions,
        "interventions": interventions,
        "outcome_counts": outcome_counts,
    }

    for name, df in tables.items():
        pq_path = processed_dir / f"{name}.parquet"
        df.to_parquet(pq_path, index=False)
        log.info("  %-20s  %6d rows → %s", name, len(df), pq_path.name)

    return tables


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print a brief summary of the prepared dataset."""
    sep = "-" * 55
    print(f"\n{sep}")
    print("  DecisionLENS — Dataset Summary")
    print(sep)

    studies = tables.get("studies")
    if studies is not None:
        print(f"  Total studies:          {len(studies):>8,}")
        if "phase" in studies.columns:
            print("  Studies by phase:")
            for phase, cnt in studies["phase"].value_counts().items():
                print(f"    {phase:<30} {cnt:>6,}")
        if "overall_status" in studies.columns:
            print("  Studies by status:")
            for status, cnt in studies["overall_status"].value_counts().items():
                print(f"    {status:<30} {cnt:>6,}")

    conditions = tables.get("conditions")
    if conditions is not None:
        print(f"  Unique conditions:      {conditions['name'].nunique():>8,}")

    facilities = tables.get("facilities")
    if facilities is not None:
        print(f"  Facility records:       {len(facilities):>8,}")
        if "country" in facilities.columns:
            print(f"  Unique countries:       {facilities['country'].nunique():>8,}")

    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DecisionLENS data setup")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("AACT_DATA_DIR", "./data")),
        help="Root data directory (default: ./data or AACT_DATA_DIR env var)",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip all other sources and generate synthetic data",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "huggingface"],
        default="auto",
        help=(
            "auto (default): try sources in priority order "
            "(processed → deployment → huggingface → local-zip → synthetic). "
            "huggingface: skip processed/deployment checks and go directly to "
            "the HuggingFace download, then fall back to synthetic."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    raw_dir        = data_dir / "raw"
    processed_dir  = data_dir / "processed"
    deployment_dir = data_dir / "deployment"

    create_directories(data_dir)

    tables: dict[str, pd.DataFrame] = {}
    source: str = ""

    if args.force_synthetic:
        # Explicit override — skip everything else
        pass

    elif args.source == "huggingface":
        # ── Direct HuggingFace mode (used by Streamlit's initialize_data) ───
        log.info("Source: huggingface (forced) — downloading from %s", HF_REPO)
        tables = download_from_huggingface(processed_dir)
        if tables:
            source = f"HuggingFace ({HF_REPO})"
        else:
            log.warning("HuggingFace download failed — falling back to synthetic.")

    else:
        # ── Auto mode: try sources in priority order ─────────────────────────

        # Priority 1: data/processed/ already contains parquet files
        if _dir_has_data(processed_dir):
            log.info("Priority 1: found existing parquet files in %s", processed_dir)
            tables = _load_parquet_tables(processed_dir)
            source = "local (data/processed)"

        # Priority 2: data/deployment/ sample (copy to processed/)
        elif _dir_has_data(deployment_dir):
            log.info(
                "Priority 2: deployment sample found in %s — copying to %s",
                deployment_dir, processed_dir,
            )
            tables = _copy_to_processed(deployment_dir, processed_dir)
            source = "local deployment sample (data/deployment)"

        # Priority 3: HuggingFace download
        else:
            log.info("Priority 3: attempting HuggingFace download …")
            tables = download_from_huggingface(processed_dir)
            if tables:
                source = f"HuggingFace ({HF_REPO})"

        # Priority 4: local AACT zip
        if not tables:
            log.info("Priority 4: looking for local AACT zip in %s …", raw_dir)
            if extract_local_zip(raw_dir):
                tables = load_raw_tables(raw_dir, processed_dir)
                if tables:
                    source = "local AACT zip"

    # ── Final fallback: synthetic data ───────────────────────────────────────
    if not tables:
        if not args.force_synthetic:
            log.warning("All data sources exhausted — generating synthetic data.")
        tables = generate_synthetic_data(processed_dir)
        source = "synthetic (fallback)"

    print_summary(tables)
    log.info("Setup complete. Source: %s  |  Files in: %s", source, processed_dir)


if __name__ == "__main__":
    main()
