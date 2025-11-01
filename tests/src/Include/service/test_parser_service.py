import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict

import settings
from Include.service.parser_service import ParserService

def make_mock_service():
	"""Helper to create ParserService with mocked wrapper"""
	service = object.__new__(ParserService)
	service._wrapper = MagicMock()
	return service

@pytest.mark.parametrize("user_input,expected", [
	("1", 0),
	("2", 1),
	("3", -1),  # skip request
	("0", -1),  # out of range
	("abc", -1), # invalid input
])
def test_handle_options_basic(monkeypatch, user_input, expected):
	options = ["opt1", "opt2"]
	monkeypatch.setattr("builtins.input", lambda _: user_input)

	service = make_mock_service()
	result = service._handle_options(options)
	
	assert result == expected

@pytest.mark.parametrize("user_input,expected", [
	("1", 0),
	("2", 1),
	("3", 2),
	("4", -1),
])
def test_handle_options_with_key(monkeypatch, user_input, expected):
	options = [1, 2, 3]
	monkeypatch.setattr("builtins.input", lambda _: user_input)

	service = make_mock_service()
	result = service._handle_options(options, key=lambda x: f"val-{x}")
	
	assert result == expected

def test_handle_options_empty(monkeypatch):
	
	options = []
	monkeypatch.setattr("builtins.input", lambda _: "1")

	service = make_mock_service()
	result = service._handle_options(options)

	assert result == -1

@pytest.mark.parametrize("test_case", [
	{
		"name": "single_option_automatic",
		"options": [(1, "test-value")],
		"expected": (1, "test-value"),
		"description": "When only one option exists, return it without user input"
	}
])
def test_handle_argument_group_options_automatic(test_case):
	"""Test cases where _handle_argument_group_options can make automatic decisions"""
	# Arrange
	service = make_mock_service()
	service._extract_argumentgroup_options = MagicMock(return_value=test_case["options"])
	service._handle_options = MagicMock()

	# Act
	result = service._handle_argument_group_options(
		action="test-action",
		argument_indices=[1],
		argument_group=(["keyword1"], {("test-value", True)}),
		max_possibilities=5
	)

	# Assert
	assert result == test_case["expected"]
	service._extract_argumentgroup_options.assert_called_once()
	service._handle_options.assert_not_called()

@pytest.mark.parametrize("test_case", [
	{
		"name": "user_selects_first",
		"options": [(1, "option1"), (2, "option2")],
		"user_choice": 0,
		"expected": (1, "option1"),
		"description": "User selects first option"
	},
	{
		"name": "user_selects_second",
		"options": [(1, "option1"), (2, "option2")],
		"user_choice": 1,
		"expected": (2, "option2"),
		"description": "User selects second option"
	},
	{
		"name": "user_skips",
		"options": [(1, "option1"), (2, "option2")],
		"user_choice": -1,
		"expected": (None, None),
		"description": "User skips selection"
	}
])
@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_handle_argument_group_options_user_input(mock_usagedb, test_case):
	"""Test cases where user input is required for selection"""
	# Arrange
	service = make_mock_service()
	service._extract_argumentgroup_options = MagicMock(return_value=test_case["options"])
	service._handle_options = MagicMock(return_value=test_case["user_choice"])
	service._wrapper.get_argument_description = MagicMock(return_value="test description")
	service._wrapper.train_argument_pipeline = MagicMock()
	
	# Act
	result = service._handle_argument_group_options(
		action="test-action",
		argument_indices=[1, 2],
		argument_group=(["keyword1", "keyword2"], {("option1", True), ("option2", False)}),
		max_possibilities=5
	)

	# Assert
	assert result == test_case["expected"]
	if test_case["user_choice"] >= 0:
		service._wrapper.train_argument_pipeline.assert_called_once()

@pytest.mark.parametrize("test_case", [
	{
		"name": "too_many_possibilities",
		"error": SyntaxError("Too many possibilities"),
		"description": "Should raise when too many options found"
	},
	{
		"name": "no_valid_arguments",
		"options": [],
		"description": "Should raise when no valid options found"
	}
])
@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_handle_argument_group_options_errors(mock_usagedb, test_case):
	"""Test error handling scenarios"""
	# Arrange
	service = make_mock_service()
	if "error" in test_case:
		service._extract_argumentgroup_options = MagicMock(side_effect=test_case["error"])
	else:
		service._extract_argumentgroup_options = MagicMock(return_value=test_case["options"])

	# Act & Assert
	with pytest.raises(SyntaxError):
		service._handle_argument_group_options(
			action="test-action",
			argument_indices=[1],
			argument_group=(["keyword1"], {("test-value", True)}),
			max_possibilities=5
		)

@patch('Include.subsystem.usagedata_db.UsagedataDB')
def test_handle_argument_group_options_training_fails(mock_usagedb):
	"""Test graceful handling of training failures"""
	# Arrange
	service = make_mock_service()
	options = [(1, "option1"), (2, "option2")]
	service._extract_argumentgroup_options = MagicMock(return_value=options)
	service._handle_options = MagicMock(return_value=0)
	service._wrapper.get_argument_description = MagicMock(return_value="test description")
	service._wrapper.train_argument_pipeline = MagicMock(side_effect=Exception("Training failed"))

	# Act
	result = service._handle_argument_group_options(
		action="test-action",
		argument_indices=[1],
		argument_group=(["keyword1"], {("option1", True)}),
		max_possibilities=5
	)

	# Assert
	assert result == options[0]  # Should return selected option despite training failure
	service._wrapper.train_argument_pipeline.assert_called_once()


@pytest.mark.parametrize("test_case", [
	{
		"name": "any_type_priority_single",
		"type": "any",
		"nonkeywords": {"int": ["1"]},
		"priority_nonkeywords": {"str": ["test"]},
		"expected": "test",
		"description": "For type 'any', should pop from priority keywords first"
	},
	{
		"name": "any_type_non_priority",
		"type": "any",
		"nonkeywords": {"int": ["1"]},
		"priority_nonkeywords": {},
		"expected": "1",
		"description": "For type 'any', fall back to non-priority if no priority exists"
	},
	{
		"name": "specific_type_priority",
		"type": "str",
		"nonkeywords": {"str": ["second"]},
		"priority_nonkeywords": {"str": ["first"]},
		"expected": "first",
		"description": "For specific type, prefer priority keywords"
	},
	{
		"name": "specific_type_non_priority",
		"type": "int",
		"nonkeywords": {"int": ["42"]},
		"priority_nonkeywords": {},
		"expected": "42",
		"description": "For specific type, use non-priority if no priority exists"
	},
])
def test_pop_nonkeyword_success(test_case):
	"""Test successful keyword popping scenarios"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	# Act
	result = service._pop_nonkeyword(
		type=test_case["type"],
		classified_nonkeywords=nonkeywords,
		classified_priority_nonkeywords=priority_nonkeywords,
		throw_if_not_found=True
	)

	# Assert
	assert result == test_case["expected"]

@pytest.mark.parametrize("test_case", [
	{
		"name": "empty_any_type",
		"type": "any",
		"nonkeywords": {},
		"priority_nonkeywords": {},
		"throw_if_not_found": True,
		"description": "Should raise when no keywords exist for type 'any'"
	},
	{
		"name": "missing_specific_type",
		"type": "str",
		"nonkeywords": {"int": ["1"]},
		"priority_nonkeywords": {"int": ["2"]},
		"throw_if_not_found": True,
		"description": "Should raise when specific type doesn't exist"
	}
])
def test_pop_nonkeyword_errors(test_case):
	"""Test error cases for keyword popping"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	# Act & Assert
	with pytest.raises(SyntaxError):
		service._pop_nonkeyword(
			type=test_case["type"],
			classified_nonkeywords=nonkeywords,
			classified_priority_nonkeywords=priority_nonkeywords,
			throw_if_not_found=test_case["throw_if_not_found"]
		)

@pytest.mark.parametrize("test_case", [
	{
		"name": "cleanup_empty_priority",
		"type": "str",
		"nonkeywords": {},
		"priority_nonkeywords": {"str": ["only"]},
		"description": "Should remove empty priority type after popping"
	},
	{
		"name": "cleanup_empty_non_priority",
		"type": "int",
		"nonkeywords": {"int": ["42"]},
		"priority_nonkeywords": {},
		"description": "Should remove empty non-priority type after popping"
	}
])
def test_pop_nonkeyword_cleanup(test_case):
	"""Test cleanup of empty containers after popping"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	# Act
	service._pop_nonkeyword(
		type=test_case["type"],
		classified_nonkeywords=nonkeywords,
		classified_priority_nonkeywords=priority_nonkeywords,
		throw_if_not_found=False
	)

	# Assert
	if "str" in test_case["priority_nonkeywords"]:
		assert test_case["type"] not in priority_nonkeywords
	if "int" in test_case["nonkeywords"]:
		assert test_case["type"] not in nonkeywords

@pytest.mark.parametrize("test_case", [
	{
		"name": "any_type_empty_not_found",
		"type": "any",
		"description": "test desc",
		"nonkeywords": {},
		"priority_nonkeywords": {},
		"throw_if_not_found": True,
		"expected_error": "Could not find 'test desc'"
	},
	{
		"name": "specific_type_missing_throw",
		"type": "str",
		"description": "test desc",
		"nonkeywords": {"int": ["1"]},
		"priority_nonkeywords": {},
		"throw_if_not_found": True,
		"expected_error": "Could not find 'test desc'"
	}
])
def test_pop_nonkeyword_question_errors(test_case):
	"""Test error cases in pop_nonkeyword_question"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	# Act & Assert
	with pytest.raises(SyntaxError) as exc_info:
		service._pop_nonkeyword_question(
			type=test_case["type"],
			description=test_case["description"],
			classified_nonkeywords=nonkeywords,
			classified_priority_nonkeywords=priority_nonkeywords,
			throw_if_not_found=test_case["throw_if_not_found"]
		)
	assert str(exc_info.value) == test_case["expected_error"]

@pytest.mark.parametrize("test_case", [
	{
		"name": "single_option_auto_return",
		"type": "str",
		"description": "test desc",
		"nonkeywords": {},
		"priority_nonkeywords": {"str": ["only_option"]},
		"expected": ("only_option", False),
		"description": "Single option should be returned without user input"
	},
	{
		"name": "user_selects_option",
		"type": "any",
		"description": "test desc",
		"nonkeywords": {"str": ["opt1", "opt2"]},
		"priority_nonkeywords": {},
		"user_choice": 1,
		"expected": ("opt2", False),
		"description": "User selects second option"
	},
	{
		"name": "user_skips",
		"type": "str",
		"description": "test desc",
		"nonkeywords": {"str": ["opt1", "opt2"]},
		"priority_nonkeywords": {},
		"user_choice": -1,
		"expected": (None, True),
		"description": "User skips selection"
	}
])
def test_pop_nonkeyword_question_success(test_case):
	"""Test successful scenarios in pop_nonkeyword_question"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	if "user_choice" in test_case:
		service._handle_options = MagicMock(return_value=test_case["user_choice"])

	# Act
	result = service._pop_nonkeyword_question(
		type=test_case["type"],
		description=test_case["description"],
		classified_nonkeywords=nonkeywords,
		classified_priority_nonkeywords=priority_nonkeywords
	)

	# Assert
	assert result == test_case["expected"]

@pytest.mark.parametrize("test_case", [
	{
		"name": "cleanup_after_selection",
		"type": "str",
		"description": "test desc",
		"nonkeywords": {"str": ["opt1"]},
		"priority_nonkeywords": {},
		"user_choice": 0,
		"expected": ("opt1", False),
		"description": "Should remove selected option and cleanup empty containers"
	}
])
def test_pop_nonkeyword_question_cleanup(test_case):
	"""Test dictionary cleanup after selection"""
	# Arrange
	service = make_mock_service()
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])
	service._handle_options = MagicMock(return_value=test_case["user_choice"])

	# Act
	result = service._pop_nonkeyword_question(
		type=test_case["type"],
		description=test_case["description"],
		classified_nonkeywords=nonkeywords,
		classified_priority_nonkeywords=priority_nonkeywords
	)

	# Assert
	assert result == test_case["expected"]
	assert test_case["type"] not in nonkeywords  # Container should be cleaned up


@pytest.mark.parametrize("test_case", [
	{
		"name": "specific_types_mapping",
		"action": "test-action",
		"argument_indices": [0, 1],
		"type_map": {"0": "str", "1": "int"},
		"nonkeywords": {"str": ["text"], "int": ["42"]},
		"priority_nonkeywords": {},
		"expected_assigned": [(0, "text"), (1, "42")],
		"expected_unassigned": [],
		"description": "Maps specific types to available non-keywords"
	},
	{
		"name": "any_type_mapping",
		"action": "test-action",
		"argument_indices": [0],
		"type_map": {"0": "any"},
		"nonkeywords": {"str": ["text"]},
		"priority_nonkeywords": {},
		"expected_assigned": [(0, "text")],
		"expected_unassigned": [],
		"description": "Maps 'any' type to available non-keywords"
	},
	{
		"name": "priority_over_regular",
		"action": "test-action",
		"argument_indices": [0],
		"type_map": {"0": "str"},
		"nonkeywords": {"str": ["regular"]},
		"priority_nonkeywords": {"str": ["priority"]},
		"expected_assigned": [(0, "priority")],
		"expected_unassigned": [],
		"description": "Prefers priority keywords over regular ones"
	},
	{
		"name": "mixed_types_with_unassigned",
		"action": "test-action",
		"argument_indices": [0, 1, 2],
		"type_map": {"0": "str", "1": "any", "2": "int"},
		"nonkeywords": {"str": ["text"], "int": []},
		"priority_nonkeywords": {},
		"expected_assigned": [(0, "text")],
		"expected_unassigned": [1, 2],
		"description": "Handles mix of assigned and unassigned arguments"
	}
])
def test_extract_arguments_typemapping_success(test_case):
	"""Test successful argument type mapping scenarios"""
	# Arrange
	service = make_mock_service()
	service._wrapper.get_argument_type = MagicMock(
		side_effect=lambda action, idx: test_case["type_map"][str(idx)]
	)
	service._wrapper.get_argument_description = MagicMock(return_value="test description")
	
	nonkeywords = defaultdict(list, test_case["nonkeywords"])
	priority_nonkeywords = defaultdict(list, test_case["priority_nonkeywords"])

	# Act
	assigned, unassigned = service._extract_arguments_typemapping(
		action=test_case["action"],
		argument_indices=test_case["argument_indices"],
		classified_nonkeywords=nonkeywords,
		classified_priority_nonkeywords=priority_nonkeywords,
		throw_if_not_found=False
	)

	# Assert
	assert assigned == test_case["expected_assigned"]
	assert unassigned == test_case["expected_unassigned"]

def test_extract_arguments_typemapping_wrapper_error():
	"""Test error handling when wrapper.get_argument_type fails"""
	# Arrange
	service = make_mock_service()
	service._wrapper.get_argument_type = MagicMock(side_effect=RuntimeError("Failed to get type"))

	# Act & Assert
	with pytest.raises(RuntimeError) as exc_info:
		service._extract_arguments_typemapping(
			action="test-action",
			argument_indices=[0],
			classified_nonkeywords={},
			classified_priority_nonkeywords={},
			throw_if_not_found=False
		)
	assert "Failed to get type" in str(exc_info.value)

def test_extract_arguments_typemapping_required_not_found():
	"""Test error when required type is not found"""
	# Arrange
	service = make_mock_service()
	service._wrapper.get_argument_type = MagicMock(return_value="str")
	service._wrapper.get_argument_description = MagicMock(return_value="test description")
	nonkeywords = defaultdict(list, {"int": ["42"]})
	priority_nonkeywords = defaultdict(list)

	# Act & Assert
	with pytest.raises(SyntaxError) as exc_info:
		service._extract_arguments_typemapping(
			action="test-action",
			argument_indices=[0],
			classified_nonkeywords=nonkeywords,
			classified_priority_nonkeywords=priority_nonkeywords,
			throw_if_not_found=True
		)
	assert "Could not find valid value" in str(exc_info.value)

@pytest.mark.parametrize("test_case", [
	{
		"name": "any_type_priority_nonkeywords",
		"action": "test-action",
		"arg_type": "any",
		"argument_indices": [0],
		"non_keywords": [("test1", False), ('1', True)],
		"expected_options": [(0, '1')],
		"description": "For type 'any', should use priority non-keywords if available"
	},
	{
		"name": "any_type_regular_nonkeywords",
		"action": "test-action",
		"arg_type": "any",
		"argument_indices": [0],
		"non_keywords": [("test1", False), ('1', False)],
		"expected_options": [(0, "test1"), (0, '1')],
		"description": "For type 'any', use regular non-keywords if no priority ones exist"
	},
	{
		"name": "specific_type_priority",
		"action": "test-action",
		"arg_type": "str",
		"argument_indices": [0],
		"non_keywords": [("test", False), ("tests", True)],
		"expected_options": [(0, "tests")],
		"description": "For specific type, should use priority non-keywords of matching type"
	},
	{
		"name": "specific_type_regular",
		"action": "test-action",
		"arg_type": "str",
		"argument_indices": [0],
		"non_keywords": [("test", False), ('1', False)],
		"expected_options": [(0, "test")],
		"description": "For specific type, should use regular non-keywords of matching type"
	},
])
def test_extract_argumentgroup_options_success(test_case):
	"""Test successful scenarios for argument group option extraction"""
	# Arrange
	service = make_mock_service()
	service._wrapper.get_argument_type = MagicMock(return_value=test_case["arg_type"])

	# Act
	result = service._extract_argumentgroup_options(
		action=test_case["action"],
		argument_indices=test_case["argument_indices"],
		non_keywords=test_case["non_keywords"]
	)

	# Assert
	assert sorted(result) == sorted(test_case["expected_options"])

@pytest.mark.parametrize("test_case", [
	{
		"name": "exceed_max_possibilities",
		"action": "test-action",
		"arg_type": "any",
		"argument_indices": [0, 1, 2],
		"non_keywords": [("test1", False), ("test2", False), ("test3", False)],
		"throw_count": 2,
		"expected_error": "Too many possibilities",
		"description": "Should raise error when possibilities exceed throw_count"
	}
])
def test_extract_argumentgroup_options_errors(test_case):
	"""Test error scenarios for argument group option extraction"""
	# Arrange
	service = make_mock_service()
	service._wrapper.get_argument_type = MagicMock(return_value=test_case["arg_type"])

	# Act & Assert
	with pytest.raises(SyntaxError) as exc_info:
		service._extract_argumentgroup_options(
			action=test_case["action"],
			argument_indices=test_case["argument_indices"],
			non_keywords=test_case["non_keywords"],
			throw_if_exceed_count=test_case["throw_count"]
		)
	assert test_case["expected_error"] in str(exc_info.value)

@pytest.mark.parametrize("test_case", [
	{
		"name": "user_declines",
		"app": "chrome",
		"handle_options_return": 1,
		"description": "User declines to set nickname, no nickname should be set"
	},
	{
		"name": "user_accepts",
		"app": "firefox",
		"handle_options_return": 0,
		"nickname": "ff",
		"description": "User accepts and sets nickname for the app"
	},
	{
		"name": "user_skips",
		"app": "vscode",
		"handle_options_return": -1,
		"description": "User skips nickname option, no nickname should be set"
	}
])
def test_handle_nickname_success(monkeypatch, test_case):
    """Test nickname handling with different user responses"""
    # Arrange
    service = make_mock_service()
    service._handle_options = MagicMock(return_value=test_case["handle_options_return"])
    service._wrapper.set_nickname = MagicMock()
    
    if test_case["handle_options_return"] == 0:
        # Only set up input mock if user accepts nickname option
        monkeypatch.setattr("builtins.input", lambda _: test_case["nickname"])

    # Act
    service._handle_nickname(test_case["app"])

    # Assert
    service._handle_options.assert_called_once_with(
        ["Yes", "No"], 
        options_message=f"Do you want to set a nickname for '{test_case['app']}'?"
    )
    
    if test_case["handle_options_return"] == 0:
        # Verify set_nickname was called with correct args if user accepted
        service._wrapper.set_nickname.assert_called_once_with(
            test_case["nickname"], 
            test_case["app"]
        )
    else:
        # Verify set_nickname was not called if user declined/skipped
        service._wrapper.set_nickname.assert_not_called()

@pytest.mark.parametrize("test_case", [
    {
        "name": "user_declines_class_assignment",
        "app": "chrome",
        "first_answer": 1,  # selects "No"
        "expected_calls": [],
        "description": "User chooses not to add the app to a class"
    },
    {
        "name": "user_selects_existing_class",
        "app": "firefox",
        "first_answer": 0,  # selects "Yes"
        "second_answer": 0,  # selects first existing class
        "classes": ["Work", "Personal"],
        "expected_class": "Work",
        "description": "User adds the app to an existing class"
    },
    {
        "name": "user_creates_new_class",
        "app": "vscode",
        "first_answer": 0,  # selects "Yes"
        "second_answer": 2,  # selects "Create new class"
        "classes": ["Work", "Personal"],
        "input_class_name": "Dev Tools",
        "expected_class": "Dev Tools",
        "description": "User chooses to create a new class"
    },
    {
        "name": "user_skips_invalid_option",
        "app": "notepad",
        "first_answer": 0,
        "second_answer": 99,  # invalid index
        "classes": ["Office"],
        "expected_calls": [],
        "description": "User provides invalid class index, no class is added"
    }
])
def test_handle_class(monkeypatch, test_case):
    """Test different user interaction flows in _handle_class"""
    # Arrange
    service = make_mock_service()
    service._handle_options = MagicMock(side_effect=[
        test_case["first_answer"],
        test_case.get("second_answer", None)
    ])
    service._wrapper.get_classes = MagicMock(return_value=test_case.get("classes", []))
    service._wrapper.add_to_class = MagicMock()

    # Mock user input when creating new class
    if "input_class_name" in test_case:
        monkeypatch.setattr("builtins.input", lambda _: test_case["input_class_name"])

    # Act
    service._handle_class(test_case["app"])

    # Assert
    service._handle_options.assert_any_call(
        ["Yes", "No"],
        options_message=f"Do you want to add '{test_case['app']}' to a class?"
    )

    if test_case["first_answer"] != 0:
        # User declined — should not proceed further
        service._wrapper.add_to_class.assert_not_called()
        return

    service._handle_options.assert_any_call(
        [*test_case.get("classes", []), "Create new class"],
        options_message=f"Do you want to add '{test_case['app']}' to an existing class, or create a new class?"
    )

    # Verify expected behavior based on scenario
    if "expected_class" in test_case:
        service._wrapper.add_to_class.assert_called_once_with(
            test_case["expected_class"], test_case["app"]
        )
    else:
        service._wrapper.add_to_class.assert_not_called()