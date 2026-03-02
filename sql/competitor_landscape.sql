/*
 * competitor_landscape.sql
 * ---------------------------------------------------------------------------
 * Competitive landscape analysis for a given condition and date window.
 *
 * Usage: Replace the parameter block below with your condition/dates of
 *        interest before running, or call via psql \set variables:
 *
 *   psql ... -v condition="'type 2 diabetes'" \
 *            -v window_start="'2022-01-01'" \
 *            -v window_end="'2024-12-31'"
 *
 * Output: One row per competing trial with competition metrics,
 *         plus a summary rollup at the end via UNION.
 *
 * Requires: AACT database — https://aact.ctti-clinicaltrials.org/
 * ---------------------------------------------------------------------------
 */

-- ============================================================
-- Parameters — edit these to target a specific query
-- ============================================================
DO $$
BEGIN
    -- These are set here for documentation; override with \set in psql.
    NULL;
END $$;

-- Use psql \set or inline literals:
--   :condition    e.g. 'type 2 diabetes'
--   :window_start e.g. '2020-01-01'
--   :window_end   e.g. '2024-12-31'

-- ============================================================
-- CTE 1: All trials matching the condition of interest
-- ============================================================
WITH condition_trials AS (
    SELECT DISTINCT
        c.nct_id
    FROM ctgov.conditions c
    WHERE c.downcase_name ILIKE '%' || :condition || '%'
),

-- ============================================================
-- CTE 2: Active/competing trial base
-- Filter to trials that are recruiting or overlap the window
-- ============================================================
competing_base AS (
    SELECT
        s.nct_id,
        s.brief_title,
        s.overall_status,
        s.phase,
        s.enrollment                            AS target_enrollment,
        s.start_date,
        s.completion_date,
        s.primary_completion_date,
        EXTRACT(YEAR FROM s.start_date)         AS start_year

    FROM ctgov.studies s
    INNER JOIN condition_trials ct
        ON s.nct_id = ct.nct_id

    WHERE s.study_type = 'Interventional'
      AND s.overall_status IN (
          'Recruiting',
          'Active, not recruiting',
          'Not yet recruiting',
          'Enrolling by invitation',
          'Completed'
      )
      -- Temporal overlap: trial window intersects query window
      AND (
          s.start_date      <= :window_end::DATE
          AND COALESCE(s.completion_date, s.primary_completion_date, CURRENT_DATE + INTERVAL '5 years')
              >= :window_start::DATE
      )
),

-- ============================================================
-- CTE 3: Lead sponsors for competing trials
-- ============================================================
competing_sponsors AS (
    SELECT
        sp.nct_id,
        sp.name                                 AS sponsor_name,
        sp.agency_class
    FROM ctgov.sponsors sp
    WHERE sp.lead_or_collaborator = 'lead'
),

-- ============================================================
-- CTE 4: Site countries for each competing trial
-- (used for geographic overlap calculation)
-- ============================================================
trial_countries AS (
    SELECT
        nct_id,
        ARRAY_AGG(DISTINCT name ORDER BY name)  AS countries,
        COUNT(DISTINCT name)                    AS n_countries
    FROM ctgov.countries
    WHERE removed = 'f'
    GROUP BY nct_id
),

-- ============================================================
-- CTE 5: Sponsor-level market share in this condition
-- ============================================================
sponsor_market AS (
    SELECT
        cs.sponsor_name,
        cs.agency_class,
        COUNT(DISTINCT cb.nct_id)               AS n_trials,
        SUM(cb.target_enrollment)               AS total_target_enrollment,
        ROUND(
            COUNT(DISTINCT cb.nct_id)::NUMERIC /
            NULLIF(SUM(COUNT(DISTINCT cb.nct_id)) OVER (), 0) * 100, 2
        )                                       AS market_share_pct
    FROM competing_base cb
    LEFT JOIN competing_sponsors cs ON cb.nct_id = cs.nct_id
    GROUP BY cs.sponsor_name, cs.agency_class
),

-- ============================================================
-- CTE 6: Temporal density — trials starting each year
-- ============================================================
temporal_density AS (
    SELECT
        start_year,
        COUNT(*)                                AS new_trials_started,
        SUM(target_enrollment)                  AS new_enrollment_targets,
        SUM(COUNT(*)) OVER (
            ORDER BY start_year
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                       AS cumulative_trials
    FROM competing_base
    GROUP BY start_year
),

-- ============================================================
-- CTE 7: Per-trial competition context
-- How many other trials were recruiting at the same time?
-- (self-join on date overlap)
-- ============================================================
concurrent_counts AS (
    SELECT
        a.nct_id,
        COUNT(b.nct_id) - 1                     AS concurrent_trial_count
    FROM competing_base a
    JOIN competing_base b
        ON a.nct_id != b.nct_id
        AND a.start_date <= COALESCE(
                b.completion_date,
                b.primary_completion_date,
                CURRENT_DATE + INTERVAL '5 years'
            )
        AND COALESCE(
                a.completion_date,
                a.primary_completion_date,
                CURRENT_DATE + INTERVAL '5 years'
            ) >= b.start_date
    GROUP BY a.nct_id
),

-- ============================================================
-- CTE 8: Phase distribution among competitors
-- ============================================================
phase_dist AS (
    SELECT
        phase,
        COUNT(*)                                AS n_trials,
        SUM(target_enrollment)                  AS total_enrollment,
        ROUND(AVG(target_enrollment), 0)        AS avg_enrollment_per_trial,
        RANK() OVER (ORDER BY COUNT(*) DESC)    AS phase_rank
    FROM competing_base
    GROUP BY phase
)

-- ============================================================
-- RESULT A: Per-trial detail
-- ============================================================
SELECT
    'trial_detail'                              AS result_type,
    cb.nct_id,
    cb.brief_title,
    cb.overall_status,
    cb.phase,
    cb.target_enrollment,
    cb.start_date,
    cb.completion_date,
    cb.start_year,
    cs.sponsor_name,
    cs.agency_class                             AS sponsor_type,
    tc.n_countries,
    tc.countries                                AS country_list,
    COALESCE(cc.concurrent_trial_count, 0)      AS concurrent_competitors,

    -- Enrollment competition pressure at time of this trial
    SUM(cb.target_enrollment) OVER ()           AS total_market_enrollment_target,
    COUNT(*) OVER ()                            AS total_competing_trials,

    -- Relative enrollment share
    ROUND(
        cb.target_enrollment::NUMERIC /
        NULLIF(SUM(cb.target_enrollment) OVER (), 0) * 100, 2
    )                                           AS this_trial_enrollment_share_pct,

    -- Temporal position (trial start rank, newest = 1)
    RANK() OVER (ORDER BY cb.start_date DESC)   AS recency_rank

FROM competing_base cb
LEFT JOIN competing_sponsors cs ON cb.nct_id = cs.nct_id
LEFT JOIN trial_countries    tc ON cb.nct_id = tc.nct_id
LEFT JOIN concurrent_counts  cc ON cb.nct_id = cc.nct_id

ORDER BY cb.start_date DESC;


-- ============================================================
-- RESULT B: Landscape summary by sponsor
-- ============================================================
SELECT
    'sponsor_summary'                           AS result_type,
    sponsor_name,
    agency_class,
    n_trials,
    total_target_enrollment,
    market_share_pct
FROM sponsor_market
ORDER BY n_trials DESC
LIMIT 20;


-- ============================================================
-- RESULT C: Temporal density (trials per year)
-- ============================================================
SELECT
    'temporal_density'                          AS result_type,
    start_year::TEXT                            AS dimension,
    new_trials_started,
    new_enrollment_targets,
    cumulative_trials
FROM temporal_density
ORDER BY start_year;


-- ============================================================
-- RESULT D: Phase distribution
-- ============================================================
SELECT
    'phase_distribution'                        AS result_type,
    phase                                       AS dimension,
    n_trials,
    total_enrollment,
    avg_enrollment_per_trial,
    phase_rank
FROM phase_dist
ORDER BY phase_rank;
