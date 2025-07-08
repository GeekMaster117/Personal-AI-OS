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