from tinydb import TinyDB, Query
from datetime import datetime
from pathlib import Path

metadata_dir = (Path(__file__) / "../../metadata/").resolve()

class MetadataDB:
    def init_db(self, db: TinyDB):
        if len(db) != 0:
            return
        
        self.db.insert_multiple([
            {"type": "apps", "data": {}},
            {"type": "web", "data": {}},
            {"type": "github", "data": {}},
            {"type": "input", "data": {}}
        ])

    def handle_db(self):
        db_path = metadata_dir / (datetime.today().date().isoformat() + ".json")

        if hasattr(self, 'db'):
            if self.current_path == db_path:
                return
            self.db.close()

        self.db = TinyDB(db_path)
        self.current_path = db_path

        self.init_db(self.db)

    def __init__(self):
        metadata_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = ''

        self.handle_db()

    def load_app_data(self):
        self.handle_db()

        self.apps = self.db.get(Query().type == "apps")
        if not self.apps:
            self.apps = {"type": "apps", "data": {}}
            self.db.insert(self.apps)

    def update_app(self, name, title, duration, focus_time, focus_count):
        if not hasattr(self, 'apps'):
            print("App data not loaded. Call load_app_data() first.")
            return

        app_data = self.apps['data']
        if name not in app_data:
            app_data[name] = {
                "titles": {},
                "duration": 0,
                "focus_time": 0,
                "focus_count": 0
            }

        if title not in app_data[name]["titles"]:
            app_data[name]["titles"][title] = {
                "duration": 0,
                "focus_time": 0,
                "focus_count": 0
            }
        
        app_data[name]["duration"] += duration
        app_data[name]["focus_time"] += focus_time
        app_data[name]["focus_count"] += focus_count

        app_data[name]["titles"][title]["duration"] += duration
        app_data[name]["titles"][title]["focus_time"] += focus_time
        app_data[name]["titles"][title]["focus_count"] += focus_count

    def save_app_data(self):
        if not hasattr(self, 'apps'):
            print("App data not loaded. Call load_app_data() first.")
            return

        self.db.update({"data": self.apps['data']}, Query().type == "apps")
        
        del self.apps