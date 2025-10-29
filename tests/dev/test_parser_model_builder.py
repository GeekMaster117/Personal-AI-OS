import os
import pytest
import json
from pathlib import Path

from dev.parser_model_builder import (
    load_commands,
    clear_directory,
    ensure_parents,
    throw_if_not_valid,
    add_keywords,
    make_keywordmaps_pipelines
)

@pytest.fixture
def sample_commands():
    return {
        "test_action": {
            "keywords": ["test", "keyword"],
            "args": [{
                "keywords": ["arg", "test"],
                "type": "string",
                "format": "text",
                "required": True,
                "description": "Test argument"
            }],
            "description": "Test action",
            "warning": False
        }
    }

@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path / "test_dir")

def test_load_commands(tmp_path, sample_commands):
    # Arrange
    commands_file = tmp_path / "commands.json"
    with open(commands_file, "w") as f:
        json.dump(sample_commands, f)
    
    # Act
    loaded_commands = load_commands(str(commands_file))
    
    # Assert
    assert loaded_commands == sample_commands

def test_load_commands_invalid_json(tmp_path):
    # Arrange
    commands_file = tmp_path / "invalid.json"
    with open(commands_file, "w") as f:
        f.write("invalid json")
    
    # Act & Assert
    with pytest.raises(RuntimeError):
        load_commands(str(commands_file))

def test_load_commands_missing_action_key(tmp_path):
    # Arrange
    invalid_commands = {
        "test_action": {
            "keywords": ["test"],
            # missing args key
            "description": "Test",
            "warning": False
        }
    }
    commands_file = tmp_path / "commands.json"
    with open(commands_file, "w") as f:
        json.dump(invalid_commands, f)
    
    # Act & Assert
    with pytest.raises(SyntaxError):
        load_commands(str(commands_file))

def test_clear_directory(temp_dir):
    # Arrange
    os.makedirs(temp_dir)
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("test")
    
    # Act
    clear_directory(temp_dir)
    
    # Assert
    assert not Path(temp_dir).exists()

def test_ensure_parents(temp_dir):
    # Arrange
    nested_path = Path(temp_dir) / "nested" / "path" / "file.txt"
    
    # Act
    result = ensure_parents(str(nested_path))
    
    # Assert
    assert Path(result).parent.exists()
    assert str(nested_path) == result

def test_throw_if_not_valid():
    # Arrange
    structure = {"key1": "value1", "key2": "value2"}
    required_keys = {"key1", "key2", "key3"}
    
    # Act & Assert
    with pytest.raises(SyntaxError):
        throw_if_not_valid(structure, required_keys, "test_structure")

def test_add_keywords():
    # Arrange
    keyword_map = {}
    keywords = ["test1", "test2"]
    value = "test_value"
    
    # Act
    add_keywords(keyword_map, keywords, value)
    
    # Assert
    assert keyword_map == {
        "test1": {"test_value"},
        "test2": {"test_value"}
    }

def test_make_keywordmaps_pipelines(sample_commands):
    # Act
    keyword_action_map, keyword_argument_maps, action_pipeline, argument_pipelines = make_keywordmaps_pipelines(sample_commands)
    
    # Assert
    assert "test" in keyword_action_map
    assert "keyword" in keyword_action_map
    assert "test_action" in keyword_argument_maps
    assert action_pipeline is not None
    assert "test_action" in argument_pipelines

def test_make_keywordmaps_pipelines_empty_commands():
    # Act
    keyword_action_map, keyword_argument_maps, action_pipeline, argument_pipelines = make_keywordmaps_pipelines({})
    
    # Assert
    assert keyword_action_map == {}
    assert keyword_argument_maps == {}
    assert action_pipeline is not None
    assert argument_pipelines == {}