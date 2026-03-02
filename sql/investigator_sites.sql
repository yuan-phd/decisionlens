/*
 * investigator_sites.sql
 * ---------------------------------------------------------------------------
 * Extracts facility- and country-level enrollment performance metrics from
 * the AACT database.  Ranks sites by historical trial volume, completion
 * rates, and enrollment efficiency — the inputs needed by the
 * InvestigatorAnalyzer module.
 *
 * Output sections:
 *   A) Site-level performance (one row per facility/condition combination)
 *   B) Country-level aggregates
 *   C) Site co-participation network edges (sites that share trials)
 *
 * Requires: AACT database — https://aact.ctti-clinicaltrials.org/
 * ---------------------------------------------------------------------------
 */

-- ============================================================
-- CTE 1: Completed interventional studies with enrollment data
-- ============================================================
WITH completed_studies AS (
    SELECT
        s.nct_id,
        s.overall_status,
        s.phase,
        s.enrollment                            AS target_enrollment,
        s.start_date,
        s.completion_date,
        cv.actual_duration                      AS duration_days,
        cv.number_of_facilities,

        -- Enrollment performance ratio (capped at 2 to avoid outlier dominance)
        LEAST(
            s.enrollment::FLOAT / NULLIF(s.enrollment, 0),
            2.0
        )                                       AS enrollment_ratio,

        CASE
            WHEN s.overall_status = 'Completed' THEN 1 ELSE 0
        END                                     AS completed_flag

    FROM ctgov.studies s
    LEFT JOIN ctgov.calculated_values cv
        ON s.nct_id = cv.nct_id

    WHERE s.study_type = 'Interventional'
      AND s.start_date >= '2005-01-01'
      AND s.enrollment  > 0
),

-- ============================================================
-- CTE 2: Primary condition per study
-- ============================================================
primary_condition AS (
    SELECT
        nct_id,
        downcase_name                           AS condition
    FROM (
        SELECT
            nct_id,
            downcase_name,
            ROW_NUMBER() OVER (PARTITION BY nct_id ORDER BY id) AS rn
        FROM ctgov.conditions
    ) ranked
    WHERE rn = 1
),

-- ============================================================
-- CTE 3: Facilities joined to their studies
-- Each row = one facility participating in one study
-- ============================================================
facility_study AS (
    SELECT
        f.nct_id,
        f.name                                  AS facility_name,
        f.city,
        f.state,
        f.country,
        f.status                                AS facility_status,
        pc.condition,
        cs.target_enrollment,
        cs.duration_days,
        cs.enrollment_ratio,
        cs.completed_flag,
        cs.phase,
        cs.start_date,
        cs.overall_status

    FROM ctgov.facilities f
    INNER JOIN completed_studies cs
        ON f.nct_id = cs.nct_id
    LEFT JOIN primary_condition pc
        ON f.nct_id = pc.nct_id

    WHERE f.status != 'Withdrawn'
      AND f.country IS NOT NULL
),

-- ============================================================
-- CTE 4: Site-level performance aggregates
-- Group by facility (name + city + country) and condition
-- ============================================================
site_performance AS (
    SELECT
        facility_name,
        city,
        country,
        condition,

        COUNT(DISTINCT nct_id)                  AS n_trials,
        SUM(completed_flag)                     AS n_completed,
        ROUND(
            SUM(completed_flag)::NUMERIC /
            NULLIF(COUNT(DISTINCT nct_id), 0) * 100, 1
        )                                       AS completion_rate_pct,

        ROUND(AVG(enrollment_ratio)::NUMERIC, 3)
                                                AS avg_enrollment_ratio,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY enrollment_ratio
        )::NUMERIC, 3)                          AS median_enrollment_ratio,

        ROUND(AVG(duration_days)::NUMERIC, 0)   AS avg_duration_days,
        ROUND(AVG(target_enrollment)::NUMERIC, 0)
                                                AS avg_target_enrollment,

        -- Phase mix
        STRING_AGG(DISTINCT phase, ', ' ORDER BY phase)
                                                AS phases_participated,

        MIN(start_date)                         AS first_trial_date,
        MAX(start_date)                         AS most_recent_trial_date,

        -- Recent activity indicator (any trial started in last 5 years)
        MAX(CASE WHEN start_date >= CURRENT_DATE - INTERVAL '5 years'
                 THEN 1 ELSE 0 END)             AS is_recently_active

    FROM facility_study
    GROUP BY facility_name, city, country, condition
),

-- ============================================================
-- CTE 5: Global site ranking (across all conditions)
-- Window rank by trial volume and completion rate
-- ============================================================
site_ranked AS (
    SELECT
        *,
        RANK() OVER (
            PARTITION BY condition
            ORDER BY n_trials DESC, completion_rate_pct DESC
        )                                       AS rank_in_condition,

        RANK() OVER (
            ORDER BY n_trials DESC, completion_rate_pct DESC
        )                                       AS global_rank,

        -- Composite performance score (0–100)
        ROUND(
            (   0.4 * LEAST(completion_rate_pct, 100)
              + 0.4 * LEAST(avg_enrollment_ratio * 50, 50)
              + 0.2 * LEAST(n_trials::FLOAT / 10, 10) * 10
            ), 1
        )                                       AS performance_score
    FROM site_performance
),

-- ============================================================
-- CTE 6: Country-level aggregates
-- ============================================================
country_performance AS (
    SELECT
        fs.country,
        fs.condition,

        COUNT(DISTINCT fs.nct_id)               AS n_trials,
        COUNT(DISTINCT
            CONCAT(fs.facility_name, '|', fs.city)
        )                                       AS n_unique_sites,
        ROUND(AVG(fs.enrollment_ratio)::NUMERIC, 3)
                                                AS avg_enrollment_ratio,
        ROUND(AVG(fs.duration_days)::NUMERIC, 0)
                                                AS avg_duration_days,
        ROUND(
            SUM(fs.completed_flag)::NUMERIC /
            NULLIF(COUNT(DISTINCT fs.nct_id), 0) * 100, 1
        )                                       AS trial_completion_rate_pct,

        -- Enrollment efficiency: avg enrollment ratio / avg duration (normalised)
        ROUND(
            AVG(fs.enrollment_ratio) /
            NULLIF(AVG(fs.duration_days) / 365.0, 0), 4
        )                                       AS enrollment_efficiency,

        RANK() OVER (
            PARTITION BY fs.condition
            ORDER BY AVG(fs.enrollment_ratio) DESC
        )                                       AS rank_in_condition,

        RANK() OVER (
            ORDER BY COUNT(DISTINCT fs.nct_id) DESC
        )                                       AS global_volume_rank

    FROM facility_study fs
    GROUP BY fs.country, fs.condition
),

-- ============================================================
-- CTE 7: Site co-participation network edges
-- Two sites are connected if they appear in the same trial.
-- Limited to pairs with at least 2 shared trials to reduce noise.
-- ============================================================
site_pairs AS (
    SELECT
        a.facility_name                         AS site_a,
        a.city                                  AS city_a,
        a.country                               AS country_a,
        b.facility_name                         AS site_b,
        b.city                                  AS city_b,
        b.country                               AS country_b,
        COUNT(DISTINCT a.nct_id)                AS shared_trials,
        a.condition

    FROM facility_study a
    JOIN facility_study b
        ON a.nct_id    = b.nct_id
        AND a.country  = b.country              -- limit to within-country edges
        AND (a.facility_name, a.city) < (b.facility_name, b.city)  -- avoid duplicates
    GROUP BY
        a.facility_name, a.city, a.country,
        b.facility_name, b.city, b.country,
        a.condition
    HAVING COUNT(DISTINCT a.nct_id) >= 2        -- only meaningful relationships
)

-- ============================================================
-- RESULT A: Site-level performance table
-- ============================================================
SELECT
    'site_performance'                          AS result_type,
    facility_name,
    city,
    country,
    condition,
    n_trials,
    n_completed,
    completion_rate_pct,
    avg_enrollment_ratio,
    median_enrollment_ratio,
    avg_duration_days,
    avg_target_enrollment,
    phases_participated,
    first_trial_date,
    most_recent_trial_date,
    is_recently_active,
    rank_in_condition,
    global_rank,
    performance_score
FROM site_ranked
ORDER BY condition, rank_in_condition;


-- ============================================================
-- RESULT B: Country-level performance
-- ============================================================
SELECT
    'country_performance'                       AS result_type,
    country,
    condition,
    n_trials,
    n_unique_sites,
    avg_enrollment_ratio,
    avg_duration_days,
    trial_completion_rate_pct,
    enrollment_efficiency,
    rank_in_condition,
    global_volume_rank
FROM country_performance
ORDER BY condition, rank_in_condition;


-- ============================================================
-- RESULT C: Site network edges (co-participation)
-- ============================================================
SELECT
    'site_network_edge'                         AS result_type,
    site_a,
    city_a,
    country_a,
    site_b,
    city_b,
    country_b,
    shared_trials,
    condition
FROM site_pairs
ORDER BY condition, shared_trials DESC;
