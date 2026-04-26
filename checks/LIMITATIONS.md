## Data Limitations

Fields not available in current AACT parquet schema:

Check A (temporal):
- results_first_submitted: not in studies.parquet
- last_update_posted: not in studies.parquet
→ These checks would be implemented when connected to
  full AACT PostgreSQL or live EDC/CTMS data source

Check C (status):
- last_update_posted: not in studies.parquet
→ Stale recruiting check uses study age proxy instead

Check D (endpoints):
- outcome timeframe: not in current parquet schema
- arm count per group: not in current parquet schema
→ These fields exist in full AACT schema (outcome_measures,
  design_groups tables) — trivial to add with full connection
- study_type granularity: studies.parquet has only "Interventional"
  for all rows; cannot cleanly distinguish observational / registry
  studies. Phase = "N/A" or null is used as a proxy for
  non-traditional designs (device, behavioural), and
  missing_primary_outcome is downgraded to LOW for those trials
  rather than flagged HIGH. Full AACT exposes richer study_type
  values that would replace this proxy.

Check E (crossfield):
- adverse_events_reported flag: not directly available
→ Proxied via outcome_counts table instead

Note: All limitations are data access constraints, not
architectural constraints. Same check logic applies to
live EDC data with richer schema.

## Agent Scan Limits
- MAX_SCAN_LIMIT = 200 per check per agent call
- Full 3.4M record scan not supported in demo mode
- Production deployment would use incremental scanning
  with pagination and result streaming
- Agent Planning Node caps limit at 200 and samples
  top findings by risk score when issue count is high
