from src.Include.wrapper.sqlite_wrapper import SQLiteWrapper

import settings

class UsagedataService:
    def __init__(self, usagedata_dir: str):
        self.db = SQLiteWrapper(usagedata_dir)

    def create_if_not_exists_schema(self) -> None:
        self.db.execute_file(settings.schema_dir)