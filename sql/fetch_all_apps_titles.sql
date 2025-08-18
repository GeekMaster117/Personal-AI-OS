WITH latest_day AS (
    SELECT id 
    FROM day_log 
    ORDER BY id DESC 
    LIMIT 1
)

SELECT 
    app_log.app_name,
    app_log.total_duration AS app_total_duration,
    app_log.total_focus_duration AS app_total_focus_duration,
    app_log.total_focus_count AS app_total_focus_count,
    title_log.title_name,
    title_log.total_duration AS title_total_duration,
    title_log.total_focus_duration AS title_total_focus_duration,
    title_log.total_focus_count AS title_total_focus_count
FROM app_log
JOIN latest_day ON app_log.day_log_id = latest_day.id
LEFT JOIN title_log ON app_log.day_log_id = title_log.day_log_id AND app_log.app_name = title_log.app_name