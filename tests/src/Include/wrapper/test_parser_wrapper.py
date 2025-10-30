from unittest.mock import patch, Mock, mock_open
import tempfile
import os
import joblib

import settings
import Include.filter.stop_words as stop_words
from Include.wrapper.parser_wrapper import ParserWrapper

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_load_commands(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()

    try:
        joblib.dump({
            "start": {
                "args": ["app"],
                "warning": False,
                "description": "Start an application"
            }
        }, temp_commands.name)
            
        settings.commands_dir = temp_commands.name

        parser = ParserWrapper(settings.Environment.DEV)
        commands = parser._load_commands()
        assert "start" in commands
        assert commands["start"]["args"] == ["app"]
        assert commands["start"]["warning"] is False
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_load_keyword_map(mock_usagedb):
    temp_keyword_action_map = tempfile.NamedTemporaryFile(delete=False)
    temp_keyword_action_map.close()

    try:
        joblib.dump({
            "launch": ["start"],
            "run": ["start"]
        }, temp_keyword_action_map.name)

        settings.keyword_action_map_dir = temp_keyword_action_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        keyword_map = parser._load_keyword_map()
        assert "launch" in keyword_map
        assert "run" in keyword_map
        assert "start" in keyword_map["launch"]
    finally:
        os.unlink(temp_keyword_action_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_has_action_warning(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()

    try:
        joblib.dump({
            "start": {"warning": True},
            "exit": {"warning": False}
        }, temp_commands.name)

        settings.commands_dir = temp_commands.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        assert parser.has_action_warning("start") is True
        assert parser.has_action_warning("exit") is False
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_action_description(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()

    try:
        joblib.dump({
            "start": {"description": "Start an application"}
        }, temp_commands.name)

        settings.commands_dir = temp_commands.name

        parser = ParserWrapper(settings.Environment.DEV)
        assert parser.get_action_description("start") == "Start an application"
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_action_keyword(mock_usagedb):
    temp_keyword_action_map = tempfile.NamedTemporaryFile(delete=False)
    temp_keyword_action_map.close()
    
    try:
        joblib.dump({
            "launch": ["start"],
            "run": ["start"]
        }, temp_keyword_action_map.name)

        settings.keyword_action_map_dir = temp_keyword_action_map.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        assert parser.match_action_keyword("launch", 0.8) == "launch"
        assert parser.match_action_keyword("launc", 0.8) == "launch"  # Fuzzy matching
        assert parser.match_action_keyword("xyz", 0.8) is None
    finally:
        os.unlink(temp_keyword_action_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('settings.keyword_argument_map_dir')
def test_match_argument_keyword(mock_map_dir, mock_usagedb):
    temp_keyword_argument_map = tempfile.NamedTemporaryFile(delete=False)
    temp_keyword_argument_map.close()
    
    try:
        joblib.dump({
            "called": [0],
            "name": [0]
        }, temp_keyword_argument_map.name)

        mock_map_dir.return_value = temp_keyword_argument_map.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        assert parser.match_argument_keyword("start", "called", 0.8) == "called"
        assert parser.match_argument_keyword("start", "calle", 0.8) == "called"  # Fuzzy matching
        assert parser.match_argument_keyword("start", "xyz", 0.8) is None
    finally:
        os.unlink(temp_keyword_argument_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_is_stop_word(mock_usagedb, monkeypatch):
    monkeypatch.setattr('Include.wrapper.parser_wrapper.ENGLISH_STOP_WORDS', {'the', 'please'})
    
    parser = ParserWrapper(settings.Environment.DEV)
    assert parser.is_stop_word("the", 0.8) is True
    assert parser.is_stop_word("please", 0.8) is True
    assert parser.is_stop_word("xyz123", 0.8) is False

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_required_arguments(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()
    
    try:
        joblib.dump({
            "start": {
                "args": [
                    {"name": "app", "required": True},
                    {"name": "param", "required": False}
                ]
            }
        }, temp_commands.name)

        settings.commands_dir = temp_commands.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        required_args = parser.get_required_arguments("start")
        assert required_args == [0]  # First argument is required
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_optional_arguments(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()
    
    try:
        joblib.dump({
            "start": {
                "args": [
                    {"name": "app", "required": True},
                    {"name": "param", "required": False}
                ]
            }
        }, temp_commands.name)

        settings.commands_dir = temp_commands.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        optional_args = parser.get_optional_arguments("start")
        assert optional_args == [1]
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_argument_type(mock_usagedb):
    temp_commands = tempfile.NamedTemporaryFile(delete=False)
    temp_commands.close()
    
    try:
        joblib.dump({
            "start": {
                "args": [
                    {"name": "app", "type": "str"}
                ]
            }
        }, temp_commands.name)

        settings.commands_dir = temp_commands.name
        
        parser = ParserWrapper(settings.Environment.DEV)
        arg_type = parser.get_argument_type("start", 0)
        assert arg_type == "str"
    finally:
        os.unlink(temp_commands.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB.get_applog_titlelog')
def test_get_monitored_apps_executablepaths(mock_log):
    mock_log.return_value = {
        "Chrome": {
            "executable_path": "C:\\Program Files\\Google\\Chrome\\chrome.exe"
        },
        "Firefox": {
            "executable_path": "C:\\Program Files\\Mozilla Firefox\\firefox.exe"
        }
    }

    parser = ParserWrapper(settings.Environment.DEV)
    apps = parser.get_monitored_apps_executablepaths()
    assert "Chrome" in apps
    assert apps["Chrome"] == "C:\\Program Files\\Google\\Chrome\\chrome.exe"

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_nickname(mock_usagedb):
    temp_nickname_app_map = tempfile.NamedTemporaryFile(delete=False)
    temp_nickname_app_map.close()

    try:
        joblib.dump(dict(), temp_nickname_app_map.name)

        settings.nickname_app_map_dir = temp_nickname_app_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        parser.set_nickname("gc", "Chrome")
        
        nickname_map = parser._get_nickname_app_map()
        assert "gc" in nickname_map
        assert nickname_map["gc"] == "Chrome"
    finally:
        os.unlink(temp_nickname_app_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_class(mock_usagedb):
    temp_class_app_map = tempfile.NamedTemporaryFile(delete=False)
    temp_class_app_map.close()
    
    try:
        joblib.dump(dict(), temp_class_app_map.name)

        settings.nickname_app_map_dir = temp_class_app_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        parser.add_to_class("browsers", "Chrome")
        
        class_map = parser._get_class_app_map()
        assert "browsers" in class_map
        assert "Chrome" in class_map["browsers"]
    finally:
        os.unlink(temp_class_app_map.name)

@patch('joblib.dump')
@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('os.path.exists')
def test_train_action_pipeline(mock_exists, mock_usagedb, mock_joblib_dump):
    mock_exists.return_value = True
    parser = ParserWrapper(settings.Environment.DEV)
    mock_pipeline = Mock()
    mock_pipeline.named_steps = {
        "countvectorizer": Mock(),
        "sgdclassifier": Mock()
    }
    parser._get_action_pipeline = lambda: mock_pipeline
    parser._action_pipeline = mock_pipeline
    parser._get_commands = lambda: {"start": {"args": ["app"]}, "stop": {"args": ["app"]}}
    parser.train_action_pipeline(["launch", "browser"], "start")
    mock_pipeline.named_steps["sgdclassifier"].partial_fit.assert_called_once()