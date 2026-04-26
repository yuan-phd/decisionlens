"""Demo-only drug-name overlay.

Our AACT parquet uses synthetic placeholder intervention names
("Intervention_N") that will never match real OpenFDA records, which
makes the cross-source safety check produce only LOW positives in
demo runs. This overlay maps a handful of real NCT IDs (that exist
in the parquet) to plausible real drug names so check G can surface
a realistic mix of HIGH / MEDIUM / LOW findings against live FDA
data.

The overlay is strictly a demo aid — in production the check reads
drug names from ``interventions.parquet`` directly and this module is
ignored. Keep the map small (≤10 entries) so it's obvious which
trials are synthetic overrides in any scan output.

Selection rationale (see checks/test_cross_source.py):
- NCT93408848 / NCT52260022 — zero ``number_of_sae_subjects`` paired
  with pembrolizumab / aspirin (both have tens of thousands of FDA
  reports) → fires HIGH ``zero_aes_with_fda_signals``.
- NCT17735107 / NCT55031671 / NCT97806010 — small SAE-subject counts
  in Completed trials paired with sertraline / lisinopril / metformin
  (all well-reported drugs) → fires MEDIUM
  ``incomplete_ae_profile_vs_fda`` under the 50× ratio threshold.
"""

from __future__ import annotations

DEMO_DRUG_MAP: dict[str, str] = {
    # Zero SAE subjects → HIGH path
    "NCT93408848": "pembrolizumab",
    "NCT52260022": "aspirin",
    # Small SAE subjects + Completed → MEDIUM path
    "NCT17735107": "sertraline",
    "NCT55031671": "lisinopril",
    "NCT97806010": "metformin",
}
