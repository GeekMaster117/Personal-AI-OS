from unittest.mock import patch, Mock
import pytest
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
@patch('settings.argument_pipeline_dir')
@patch('os.path.exists')
def test_load_pipeline(mock_exists, mock_argument_dir, mock_usagedb):
    temp_pipeline = tempfile.NamedTemporaryFile(delete=False)
    temp_pipeline.close()

    try:
        joblib.dump(5, temp_pipeline.name)
        mock_exists.return_value = True

        parser = ParserWrapper(settings.Environment.DEV)

        settings.action_pipeline_dir = temp_pipeline.name        
        action_pipeline = parser._load_pipeline()

        assert action_pipeline == 5

        mock_argument_dir.return_value = temp_pipeline.name
        argument_pipeline = parser._load_pipeline('test')

        assert argument_pipeline == 5
    finally:
        os.unlink(temp_pipeline.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_load_app_executablepath_map(mock_usagedb):
    temp_app_executablepath_map = tempfile.NamedTemporaryFile(delete=False)
    temp_app_executablepath_map.close()

    try:
        joblib.dump({
            "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "notepad": "C:/Windows/System32/notepad.exe"
        }, temp_app_executablepath_map.name)

        settings.app_executablepath_map_dir = temp_app_executablepath_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        app_map = parser._load_app_executablepath_map()
        assert "chrome" in app_map
        assert app_map["notepad"] == "C:/Windows/System32/notepad.exe"
    finally:
        os.unlink(temp_app_executablepath_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_load_nickname_app_map(mock_usagedb):
    temp_nickname_app_map = tempfile.NamedTemporaryFile(delete=False)
    temp_nickname_app_map.close()

    try:
        joblib.dump({
            "browser": "chrome",
            "editor": "notepad"
        }, temp_nickname_app_map.name)

        settings.nickname_app_map_dir = temp_nickname_app_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        nickname_map = parser._load_nickname_app_map()
        assert "browser" in nickname_map
        assert nickname_map["editor"] == "notepad"
    finally:
        os.unlink(temp_nickname_app_map.name)


@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_load_class_app_map(mock_usagedb):
    temp_class_app_map = tempfile.NamedTemporaryFile(delete=False)
    temp_class_app_map.close()

    try:
        joblib.dump({
            "browsers": ["chrome", "firefox"],
            "editors": ["notepad", "vim"]
        }, temp_class_app_map.name)

        settings.class_app_map_dir = temp_class_app_map.name

        parser = ParserWrapper(settings.Environment.DEV)
        class_map = parser._load_class_app_map()
        assert "browsers" in class_map
        assert "editors" in class_map
        assert "vim" in class_map["editors"]
    finally:
        os.unlink(temp_class_app_map.name)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('joblib.dump')
@patch('os.path.exists')
def test_save_pipeline(mock_exists, mock_dump, mock_usagedb):
    mock_exists.return_value = True

    parser = ParserWrapper(settings.Environment.DEV)
   
    parser._save_pipeline(5)

    assert mock_dump.call_args[0][0] == 5
    assert mock_dump.call_args[0][1] == settings.action_pipeline_dir

    parser._save_pipeline(5, 'test')

    assert mock_dump.call_args[0][0] == 5
    assert mock_dump.call_args[0][1] == settings.argument_pipeline_dir('test')

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_pipeline')
def test_save_action_pipeline(mock_pipeline, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._action_pipeline = 5
    parser._save_action_pipeline()
    
    assert len(mock_pipeline.call_args[0]) == 1 or mock_pipeline.call_args[0][1] is None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_pipeline')
def test_save_argument_pipeline(mock_pipeline, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._argument_pipelines = {
        'test': 5
    }
    parser._save_argument_pipeline('test')

    assert len(mock_pipeline.call_args[0]) == 2 and mock_pipeline.call_args[0][1] is not None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('joblib.dump')
def test_save_app_executablepath_map(mock_dump, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._app_executablepath_map = {
        'test': 'test1'
    }
    parser._save_app_executablepath_map()

    assert 'test' in mock_dump.call_args[0][0]
    assert mock_dump.call_args[0][0]['test'] == 'test1'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('joblib.dump')
def test_save_nickname_app_map(mock_dump, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._nickname_app_map = {
        'test': 'test1'
    }
    parser._save_nickname_app_map()

    assert 'test' in mock_dump.call_args[0][0]
    assert mock_dump.call_args[0][0]['test'] == 'test1'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch('joblib.dump')
def test_save_class_app_map(mock_dump, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._class_app_map = {
        'test': {'test1'}
    }
    parser._save_class_app_map()

    assert 'test' in mock_dump.call_args[0][0]
    assert len(mock_dump.call_args[0][0]['test']) == 1 and 'test1' in mock_dump.call_args[0][0]['test']

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_commands')
def test_get_commands(mock_commands, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    
    parser._get_commands()

    parser._commands = dict()
    parser._get_commands()

    mock_commands.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_keyword_map')
def test_get_action_keyword_map(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_keyword_action_map()

    parser._keyword_action_map = dict()
    parser._get_keyword_action_map()

    mock_map.assert_called_once()
    assert len(mock_map.call_args[0]) == 0 or mock_map.call_args[0][0] is None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_keyword_map')
def test_get_argument_keyword_map(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_keyword_argument_map('test')

    parser._keyword_argument_maps = {
        'test': dict()
    }
    parser._get_keyword_argument_map('test')

    mock_map.assert_called_once()
    assert len(mock_map.call_args[0]) == 1 and mock_map.call_args[0][0] is not None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_pipeline')
def test_get_action_pipeline(mock_pipeline, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_action_pipeline()

    parser._action_pipeline = 5
    parser._get_action_pipeline()

    mock_pipeline.assert_called_once()
    assert len(mock_pipeline.call_args[0]) == 0 or mock_pipeline.call_args[0][0] is None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_pipeline')
def test_get_argument_pipeline(mock_pipeline, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_argument_pipeline('test')

    parser._argument_pipelines = {
        'test': dict()
    }
    parser._get_argument_pipeline('test')

    mock_pipeline.assert_called_once()
    assert len(mock_pipeline.call_args[0]) == 1 and mock_pipeline.call_args[0][0] is not None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_app_executablepath_map')
def test_get_app_executablepath_map(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_app_executablepath_map()

    parser._app_executablepath_map = dict()
    parser._get_app_executablepath_map()

    mock_map.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_nickname_app_map')
def test_get_nickname_app_map(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_nickname_app_map()

    parser._nickname_app_map = dict()
    parser._get_nickname_app_map()

    mock_map.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_load_class_app_map')
def test_get_class_app_map(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._get_class_app_map()

    parser._class_app_map = dict()
    parser._get_class_app_map()

    mock_map.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_apps_with_nicknames(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._nickname_app_map = {
        'test': 'test1'
    }
    result = parser._get_apps_with_nicknames()

    assert len(result) == 1 and 'test1' in result

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_apps_in_classes(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._class_app_map = {
        'test': {'test1', 'test2'},
        'test3': {'test4', 'test5'}
    }
    result = parser._get_apps_in_class()

    assert len(result.intersection({'test1', 'test2', 'test4', 'test5'})) == 4

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_has_nicknames(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._nickname_app_map = {
        "browser": "chrome"
    }

    assert parser.has_nicknames('chrome') is True
    assert parser.has_nicknames('firefox') is False

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_in_class(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._class_app_map = {
        "browser": {"chrome"}
    }

    assert parser.in_class('chrome') is True
    assert parser.in_class('firefox') is False

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_nickname_app_map')
def test_set_nickname(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._nickname_app_map = dict()

    parser.set_nickname("gc", "Chrome")
    result = parser._get_nickname_app_map()
    
    assert "gc" in result
    assert result["gc"] == "Chrome"

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_class_app_map')
def test_add_to_class(mock_map, mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._class_app_map = dict()

    parser.add_to_class("browsers", "Chrome")
    result = parser._get_class_app_map()

    assert "browsers" in result
    assert "Chrome" in result["browsers"]
    mock_map.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_action_pipeline')
def test_train_action_pipeline(mock_save, mock_usagedb):
    mock_pipeline = Mock()
    mock_pipeline.named_steps = {
        "countvectorizer": Mock(),
        "sgdclassifier": Mock()
    }

    parser = ParserWrapper(settings.Environment.DEV)
    parser._action_pipeline = mock_pipeline
    parser._commands = {"start": {"args": ["app"]}, "stop": {"args": ["app"]}}
    parser.train_action_pipeline(["launch", "browser"], "start")

    mock_pipeline.named_steps["sgdclassifier"].partial_fit.assert_called_once()
    mock_save.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, '_save_argument_pipeline')
def test_train_argument_pipeline(mock_save, mock_usagedb):
    mock_pipeline = Mock()
    mock_pipeline.named_steps = {
        "countvectorizer": Mock(),
        "sgdclassifier": Mock()
    }

    parser = ParserWrapper(settings.Environment.DEV)
    parser._argument_pipelines = {
        'start': mock_pipeline
    }
    parser._commands = {"start": {"args": ["app", "app1"]}, "stop": {"args": ["app", "app1"]}}
    parser.train_argument_pipeline('start', ["app", "app1"], 0)

    mock_pipeline.named_steps["sgdclassifier"].partial_fit.assert_called_once()
    mock_save.assert_called_once()
    assert mock_save.call_args[0][0] == 'start'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_predict_top_actions(mock_usagedb):
    mock_pipeline = Mock()
    mock_pipeline.predict_proba.return_value = [[0.7, 0.3]]
    mock_pipeline.classes_ = ['start', 'stop']

    parser = ParserWrapper(settings.Environment.DEV)
    parser._action_pipeline = mock_pipeline
    result = parser.predict_top_actions(['launch'], 2, 0.5)

    assert result[0][0] == 'start'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_predict_top_argument_indices(mock_usagedb):
    mock_pipeline = Mock()
    mock_pipeline.predict_proba.return_value = [[0.7, 0.3]]
    mock_pipeline.classes_ = [0, 1]

    parser = ParserWrapper(settings.Environment.DEV)
    parser._argument_pipelines = {
        'test': mock_pipeline
    }
    result = parser.predict_top_arguments_indices('test', ['launch'], 2, 0.5)

    assert result[0][0] == 0

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_predict_argument_index(mock_usagedb):
    mock_pipeline = Mock()
    mock_pipeline.predict_proba.return_value = [[0.6, 0.4]]
    mock_pipeline.classes_ = [0, 1]

    parser = ParserWrapper(settings.Environment.DEV)
    parser._get_argument_pipeline = lambda action: mock_pipeline
    idx = parser.predict_argument_index('start', ['app'], 0.5)

    assert idx == 0

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_action_keyword(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_action_map = {
        'launch': {'test'}
    }

    assert parser.match_action_keyword("launch", 0.8) == "launch"
    assert parser.match_action_keyword("launc", 0.8) == "launch"
    assert parser.match_action_keyword("xyz", 0.8) is None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_argument_keyword(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_argument_maps = {
        'start': {
            'called': {0}
        }
    }

    assert parser.match_argument_keyword("start", "called", 0.8) == "called"
    assert parser.match_argument_keyword("start", "calle", 0.8) == "called"  # Fuzzy matching
    assert parser.match_argument_keyword("start", "xyz", 0.8) is None

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_existing_app(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._app_executablepath_map = {
        'chrome': 'chrome.exe'
    }

    assert parser.match_existing_app('chrome', 0.5) == 'chrome'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
@patch.object(ParserWrapper, 'get_monitored_apps_executablepaths')
@patch.object(ParserWrapper, '_save_app_executablepath_map')
def test_match_monitored_app(mock_save, mock_monitored, mock_usagedb):
    mock_monitored.return_value = {
        'test': 'test1'
    }

    parser = ParserWrapper(settings.Environment.DEV)

    parser._app_executablepath_map = dict()

    assert parser.match_monitored_app('test', 0.5) == 'test'
    mock_save.assert_called_once()

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_nickname(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._nickname_app_map = {
        "browser": "chrome"
    }

    assert parser.match_nickname('browser', 0.5) == 'browser'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_match_class(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._class_app_map = {
        "browser": {"chrome"}
    }

    assert parser.match_class('browser', 0.5) == 'browser'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_is_stop_word(mock_usagedb, monkeypatch):
    monkeypatch.setattr('Include.wrapper.parser_wrapper.ENGLISH_STOP_WORDS', {'the', 'please'})
    
    parser = ParserWrapper(settings.Environment.DEV)
    assert parser.is_stop_word("the", 0.8) is True
    assert parser.is_stop_word("please", 0.8) is True
    assert parser.is_stop_word("xyz123", 0.8) is False

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_has_action_warning(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        "start": {"warning": True},
        "exit": {"warning": False}
    }

    assert parser.has_action_warning("start") is True
    assert parser.has_action_warning("exit") is False

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_action_description(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        "start": {"description": "Start an application"}
    }

    assert parser.get_action_description("start") == "Start an application"

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_action_keywords(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_action_map = {
        'test': {'test1'}
    }
    result = parser.get_action_keywords()

    assert len(result) == 1 and 'test' in result

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_argument_keywords(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_argument_maps = {
        'test': {
            'test1': {0}
        }
    }
    result = parser.get_argument_keywords('test')

    assert len(result) == 1 and 'test1' in result

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_actions_for_keyword_and_argument_indices_for_keyword(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_action_map = {'launch': {'start'}}
    
    assert parser.get_actions_for_keyword('launch') == {'start'}

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_argument_indices_for_keyword(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._keyword_argument_maps = {'start': {'app': {0}}}

    assert parser.get_argument_indices_for_keyword('start', 'app') == {0}

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_required_arguments(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        "start": {
            "args": [
                {"name": "app", "required": True},
                {"name": "param", "required": False}
            ]
        }
    }

    required_args = parser.get_required_arguments("start")
    assert required_args == [0]

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_optional_arguments(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        "start": {
            "args": [
                {"name": "app", "required": True},
                {"name": "param", "required": False}
            ]
        }
    }

    optional_args = parser.get_optional_arguments("start")
    assert optional_args == [1]

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_argument_type(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        'start': {
            'args': [
                {
                    'type': 'str'
                }
            ]
        }
    }

    assert parser.get_argument_type('start', 0) == 'str'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_argument_format(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    
    parser._commands = {
        'start': {
            'args': [
                {
                    'format': 'txt'
                }
            ]
        }
    }

    assert parser.get_argument_format('start', 0) == 'txt'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_argument_description(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._commands = {
        'start': {
            'args': [
                {
                    'description': 'desc'
                }
            ]
        }
    }

    assert parser.get_argument_description('start', 0) == 'desc'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_arguments_count(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)
    parser._commands = {
        'start': {
            'args': ['test', 'test1']
        }
    }

    assert parser.get_arguments_count('start') == 2

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_existing_apps(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._app_executablepath_map = {'chrome': 'path'}

    assert 'chrome' in parser.get_existing_apps()

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
def test_get_app_for_nickname(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._nickname_app_map = {'gc': 'chrome'}
    
    assert parser.get_app_for_nickname('gc') == 'chrome'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_executablepath(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._app_executablepath_map = {'chrome': 'path'}

    assert parser.get_executablepath('chrome') == 'path'

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_get_classes(mock_usagedb):
    parser = ParserWrapper(settings.Environment.DEV)

    parser._class_app_map = {'browser': {'chrome'}}

    assert 'browser' in parser.get_classes()