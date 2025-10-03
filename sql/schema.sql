CREATE TABLE IF NOT EXISTS day_log (
    id INTEGER PRIMARY KEY,
    time_anchor TEXT NOT NULL,
    monotonic_start REAL NOT NULL,
    monotonic_last_updated REAL NOT NULL,
    total_downtime_duration REAL DEFAULT 0 CHECK(total_downtime_duration >= 0),
    total_anomalies INTEGER DEFAULT 0 CHECK(total_anomalies >= 0)
);

CREATE TABLE IF NOT EXISTS app_log (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    executable_path TEXT NOT NULL,
    total_duration REAL DEFAULT 0 CHECK(total_duration >= 0),
    total_focus_duration REAL DEFAULT 0 CHECK(total_focus_duration >= 0),
    total_focus_count INTEGER DEFAULT 0 CHECK(total_focus_count >= 0),
    PRIMARY KEY(day_log_id, app_name),
    FOREIGN KEY(day_log_id) REFERENCES day_log(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS title_log (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    title_name TEXT NOT NULL,
    total_duration REAL DEFAULT 0 CHECK(total_duration >= 0),
    total_focus_duration REAL DEFAULT 0 CHECK(total_focus_duration >= 0),
    total_focus_count INTEGER DEFAULT 0 CHECK(total_focus_count >= 0),
    PRIMARY KEY(day_log_id, app_name, title_name),
    FOREIGN KEY(day_log_id, app_name) REFERENCES app_log(day_log_id, app_name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS downtime_period (
    day_log_id INTEGER NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    downtime_duration REAL DEFAULT 0 CHECK(downtime_duration BETWEEN 0 AND 3600),
    PRIMARY KEY(day_log_id, day_hour),
    FOREIGN KEY(day_log_id) REFERENCES day_log(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_focus_period (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    focus_duration REAL DEFAULT 0 CHECK(focus_duration BETWEEN 0 AND 3600),
    focus_count INTEGER DEFAULT 0 CHECK(focus_count >= 0),
    PRIMARY KEY(day_log_id, app_name, day_hour),
    FOREIGN KEY(day_log_id, app_name) REFERENCES app_log(day_log_id, app_name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS title_focus_period (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    title_name TEXT NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    focus_duration REAL DEFAULT 0 CHECK(focus_duration BETWEEN 0 AND 3600),
    focus_count INTEGER DEFAULT 0 CHECK(focus_count >= 0),
    PRIMARY KEY(day_log_id, app_name, title_name, day_hour),
    FOREIGN KEY(day_log_id, app_name, title_name) REFERENCES title_log(day_log_id, app_name, title_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_applog_daylog ON app_log(day_log_id);
CREATE INDEX IF NOT EXISTS idx_titlelog_applog ON title_log(day_log_id, app_name);
CREATE INDEX IF NOT EXISTS idx_downtimeperiod_daylog ON downtime_period(day_log_id);
CREATE INDEX IF NOT EXISTS idx_appfocusperiod_applog ON app_focus_period(day_log_id, app_name);
CREATE INDEX IF NOT EXISTS idx_titlefocusperiod_titlelog ON title_focus_period(day_log_id, app_name, title_name);