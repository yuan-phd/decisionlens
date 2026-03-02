/*
 * enrollment_extract.sql
 * ---------------------------------------------------------------------------
 * Extracts a fully-featured enrollment modeling dataset from the AACT
 * PostgreSQL schema (schema: ctgov).
 *
 * Scope   : Interventional studies with a start_date >= 2008-01-01.
 * Output  : One row per study with enrollment targets, actuals, study
 *           characteristics, site/country counts, sponsor performance, and
 *           derived labels for modeling.
 *
 * Requires: AACT database — https://aact.ctti-clinicaltrials.org/
 * Run with: psql -h aact-db.ctti-clinicaltrials.org -U <user> -d aact -f enrollment_extract.sql
 * ---------------------------------------------------------------------------
 */

-- ============================================================
-- CTE 1: Core studies — filter scope, parse key fields
-- ============================================================
WITH base_studies AS (
    SELECT
        s.nct_id,
        s.study_type,
        s.overall_status,
        s.phase,
        s.enrollment                            AS target_enrollment,
        s.start_date,
        s.completion_date,
        s.primary_completion_date,
        s.has_dmc,
        s.is_fda_regulated_drug,
        s.is_fda_regulated_device,
        s.why_stopped,
        cv.actual_duration,                     -- days from start to completion
        cv.number_of_facilities,
        cv.minimum_age_num,
        cv.maximum_age_num

    FROM ctgov.studies s
    LEFT JOIN ctgov.calculated_values cv
        ON s.nct_id = cv.nct_id

    WHERE s.study_type = 'Interventional'
      AND s.start_date  >= '2008-01-01'
      AND s.overall_status NOT IN ('Withdrawn', 'Unknown status')
      AND s.enrollment   IS NOT NULL
      AND s.enrollment   > 0
),

-- ============================================================
-- CTE 2: Primary condition (one per study, highest frequency)
-- ============================================================
conditions_ranked AS (
    SELECT
        nct_id,
        downcase_name                           AS condition,
        ROW_NUMBER() OVER (
            PARTITION BY nct_id
            ORDER BY id            -- first-listed condition
        ) AS rn
    FROM ctgov.conditions
),
primary_condition AS (
    SELECT nct_id, condition
    FROM conditions_ranked
    WHERE rn = 1
),

-- ============================================================
-- CTE 3: Condition frequency — proxy for patient pool size
-- ============================================================
condition_freq AS (
    SELECT
        downcase_name                           AS condition,
        COUNT(DISTINCT nct_id)                  AS condition_trial_count
    FROM ctgov.conditions
    GROUP BY downcase_name
),

-- ============================================================
-- CTE 4: Primary intervention type
-- ============================================================
interventions_ranked AS (
    SELECT
        nct_id,
        intervention_type,
        ROW_NUMBER() OVER (
            PARTITION BY nct_id
            ORDER BY id
        ) AS rn
    FROM ctgov.interventions
),
primary_intervention AS (
    SELECT nct_id, intervention_type
    FROM interventions_ranked
    WHERE rn = 1
),

-- ============================================================
-- CTE 5: Lead sponsor details
-- ============================================================
lead_sponsors AS (
    SELECT
        nct_id,
        name                                    AS sponsor_name,
        agency_class                            AS sponsor_agency_class,
        CASE
            WHEN agency_class = 'Industry'      THEN 'industry'
            WHEN agency_class IN ('NIH', 'U.S. Fed') THEN 'government'
            ELSE 'academic'
        END                                     AS sponsor_type
    FROM ctgov.sponsors
    WHERE lead_or_collaborator = 'lead'
),

-- ============================================================
-- CTE 6: Historical sponsor enrollment performance
-- Median (actual/target) ratio for each sponsor across past trials
-- ============================================================
sponsor_history AS (
    SELECT
        sp.name                                 AS sponsor_name,
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY s.enrollment::FLOAT / NULLIF(s.enrollment, 0)
        )                                       AS sponsor_median_enrollment_ratio,
        COUNT(*)                                AS sponsor_historical_trial_count
    FROM ctgov.sponsors sp
    JOIN ctgov.studies s
        ON sp.nct_id = s.nct_id
    WHERE sp.lead_or_collaborator = 'lead'
      AND s.overall_status = 'Completed'
      AND s.enrollment > 0
    GROUP BY sp.name
),

-- ============================================================
-- CTE 7: Number of unique countries per study
-- ============================================================
country_counts AS (
    SELECT
        nct_id,
        COUNT(DISTINCT name)                    AS n_countries
    FROM ctgov.countries
    WHERE removed = 'f'
    GROUP BY nct_id
),

-- ============================================================
-- CTE 8: Eligibility criteria complexity (line count proxy)
-- ============================================================
eligibility_complexity AS (
    SELECT
        nct_id,
        gender,
        healthy_volunteers,
        -- Count newline-delimited criteria lines as a complexity proxy
        ARRAY_LENGTH(
            STRING_TO_ARRAY(criteria, E'\n'), 1
        )                                       AS n_criteria_lines
    FROM ctgov.eligibilities
),

-- ============================================================
-- CTE 9: Study design features
-- ============================================================
design_features AS (
    SELECT
        nct_id,
        intervention_model,
        primary_purpose,
        masking,
        allocation
    FROM ctgov.designs
),

-- ============================================================
-- CTE 10: Phase ordinal encoding
-- ============================================================
phase_encoded AS (
    SELECT
        nct_id,
        phase,
        CASE phase
            WHEN 'Phase 1'          THEN 1.0
            WHEN 'Phase 1/Phase 2'  THEN 1.5
            WHEN 'Phase 2'          THEN 2.0
            WHEN 'Phase 2/Phase 3'  THEN 2.5
            WHEN 'Phase 3'          THEN 3.0
            WHEN 'Phase 4'          THEN 4.0
            ELSE NULL
        END                                     AS phase_numeric
    FROM ctgov.studies
),

-- ============================================================
-- CTE 11: Sponsor rank by historical enrollment volume
-- (window function across all sponsors)
-- ============================================================
sponsor_ranked AS (
    SELECT
        sponsor_name,
        sponsor_historical_trial_count,
        sponsor_median_enrollment_ratio,
        RANK() OVER (
            ORDER BY sponsor_historical_trial_count DESC
        )                                       AS sponsor_rank_by_volume
    FROM sponsor_history
)

-- ============================================================
-- FINAL SELECT: Assemble the modeling dataset
-- ============================================================
SELECT
    -- Identifiers
    bs.nct_id,

    -- Study characteristics
    bs.study_type,
    bs.overall_status,
    pe.phase,
    pe.phase_numeric,
    df.intervention_model,
    df.primary_purpose,
    df.masking,
    df.allocation,
    pi.intervention_type,

    -- Condition
    pc.condition,
    cf.condition_trial_count                    AS condition_prevalence_proxy,

    -- Enrollment (targets and outcomes)
    bs.target_enrollment,
    bs.actual_duration                          AS enrollment_duration_days,

    -- Site & geography
    bs.number_of_facilities                     AS n_facilities,
    COALESCE(cc.n_countries, 1)                 AS n_countries,
    CASE WHEN COALESCE(cc.n_countries, 1) > 1
         THEN TRUE ELSE FALSE END               AS is_multicountry,

    -- Age eligibility window
    bs.minimum_age_num,
    bs.maximum_age_num,
    (bs.maximum_age_num - bs.minimum_age_num)   AS age_window_years,

    -- Eligibility complexity
    ec.n_criteria_lines,
    ec.gender,
    ec.healthy_volunteers,

    -- Sponsor
    ls.sponsor_name,
    ls.sponsor_type,
    ls.sponsor_agency_class,
    COALESCE(sr.sponsor_median_enrollment_ratio, 1.0)
                                                AS sponsor_historical_performance,
    COALESCE(sr.sponsor_rank_by_volume, 9999)   AS sponsor_rank,

    -- Regulatory / oversight
    bs.has_dmc,
    bs.is_fda_regulated_drug,
    bs.is_fda_regulated_device,

    -- Dates
    bs.start_date,
    bs.completion_date,
    bs.primary_completion_date,
    EXTRACT(YEAR FROM bs.start_date)            AS start_year,
    EXTRACT(MONTH FROM bs.start_date)           AS start_month,

    -- ---- Derived modeling labels ----
    -- Binary: did trial meet ≥90% of target enrollment?
    CASE WHEN cv_outer.actual_duration > 0
              AND bs.target_enrollment > 0
         THEN CASE
                  WHEN (bs.enrollment::FLOAT / bs.target_enrollment) >= 0.9
                  THEN 1 ELSE 0
              END
         ELSE NULL
    END                                         AS enrollment_met_target,

    -- Continuous ratio for regression
    CASE WHEN bs.target_enrollment > 0
         THEN bs.enrollment::FLOAT / bs.target_enrollment
         ELSE NULL
    END                                         AS enrollment_ratio

FROM base_studies bs

-- Joins
LEFT JOIN primary_condition    pc  ON bs.nct_id = pc.nct_id
LEFT JOIN condition_freq       cf  ON pc.condition = cf.condition
LEFT JOIN primary_intervention pi  ON bs.nct_id = pi.nct_id
LEFT JOIN lead_sponsors        ls  ON bs.nct_id = ls.nct_id
LEFT JOIN sponsor_ranked       sr  ON ls.sponsor_name = sr.sponsor_name
LEFT JOIN country_counts       cc  ON bs.nct_id = cc.nct_id
LEFT JOIN eligibility_complexity ec ON bs.nct_id = ec.nct_id
LEFT JOIN design_features      df  ON bs.nct_id = df.nct_id
LEFT JOIN phase_encoded        pe  ON bs.nct_id = pe.nct_id
LEFT JOIN ctgov.calculated_values cv_outer ON bs.nct_id = cv_outer.nct_id

ORDER BY bs.start_date DESC;
