CREATE TABLE IF NOT EXISTS day_log (
    id INTEGER PRIMARY KEY,
    time_anchor TEXT NOT NULL,
    monotonic_start REAL NOT NULL,
    monotonic_last_updated REAL NOT NULL,
    total_downtime_duration REAL DEFAULT 0,
    total_anomalies INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS app (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    total_duration REAL DEFAULT 0,
    focus_duration REAL DEFAULT 0,
    focus_count INTEGER DEFAULT 0,
    PRIMARY KEY(day_log_id, app_name),
    FOREIGN KEY(day_log_id) REFERENCES day_log(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS title (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    title_name TEXT NOT NULL,
    total_duration REAL DEFAULT 0,
    focus_duration REAL DEFAULT 0,
    focus_count INTEGER DEFAULT 0,
    PRIMARY KEY(day_log_id, app_name, title_name),
    FOREIGN KEY(day_log_id, app_name) REFERENCES app(day_log_id, app_name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS downtime (
    day_log_id INTEGER NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    total_duration REAL DEFAULT CHECK(day_hour BETWEEN 0 AND 3600),
    PRIMARY KEY(day_log_id, day_hour),
    FOREIGN KEY(day_log_id) REFERENCES day_log(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_focus (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    total_duration REAL DEFAULT CHECK(day_hour BETWEEN 0 AND 3600),
    PRIMARY KEY(day_log_id, app_name, day_hour),
    FOREIGN KEY(day_log_id, app_name) REFERENCES app(day_log_id, app_name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS title_focus (
    day_log_id INTEGER NOT NULL,
    app_name TEXT NOT NULL,
    title_name TEXT NOT NULL,
    day_hour INTEGER NOT NULL CHECK(day_hour BETWEEN 0 AND 23),
    total_duration REAL DEFAULT CHECK(day_hour BETWEEN 0 AND 3600),
    PRIMARY KEY(day_log_id, app_name, title_name, day_hour),
    FOREIGN KEY(day_log_id, app_name, title_name) REFERENCES title(day_log_id, app_name, title_name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_app_daylog ON app(day_log_id);
CREATE INDEX IF NOT EXISTS idx_title_app ON title(day_log_id, app_name);
CREATE INDEX IF NOT EXISTS idx_downtime_daylog ON downtime(day_log_id);
CREATE INDEX IF NOT EXISTS idx_appfocus_app ON app_focus(day_log_id, app_name);
CREATE INDEX IF NOT EXISTS idx_titlefocus_title ON title_focus(day_log_id, app_name, title_name);