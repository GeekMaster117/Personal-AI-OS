import pytest
from unittest.mock import patch, MagicMock
from collections import defaultdict

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

@pytest.mark.parametrize("test_case", [
	{
		"name": 'warning_true_user_yes',
		"warning": True,
		"input": 'y',
		'expected': True
	},
	{
		"name": 'warning_true_user_no',
		"warning": True,
		"input": 'n',
		'expected': False
	},
	{
		"name": 'warning_true_user_random',
		"warning": True,
		"input": 'test',
		'expected': False
	},
	{
		"name": 'warning_false',
		"warning": False,
		'expected': True
	},
])
def test_can_run_action(monkeypatch, test_case):
	service = make_mock_service()
	service._wrapper.has_action_warning = MagicMock(return_value = test_case['warning'])
	monkeypatch.setattr("builtins.input", lambda _: test_case["input"])

	result = service.canrun_action('test')

	assert result == test_case['expected']

@pytest.mark.parametrize("test_case", [
	{
		"name": 'nickname_true',
		"nickname": True,
		'class': False,
		'assert': False
	},
	{
		"name": 'class_true',
		'nickname': False,
		"class": True,
		'assert': False
	},
	{
		"name": 'nickname_false_class_false',
		"nickname": False,
		'class': False,
		'assert': True
	},
	{
		"name": 'nickname_true_class_true',
		"nickname": True,
		'class': True,
		'assert': False
	}
])
def test_handle_nickname_class(test_case):
	service = make_mock_service()
	service._wrapper.has_nicknames = lambda app: test_case['nickname']
	service._wrapper.in_class = lambda app: test_case['class']

	service._handle_nickname = MagicMock()
	service._handle_class = MagicMock()

	service.handle_nickname_class('test')

	if test_case['assert']:
		service._handle_nickname.assert_called_once()
		service._handle_class.assert_called_once()
	else:
		service._handle_nickname.assert_not_called()
		service._handle_class.assert_not_called()

@pytest.mark.parametrize("test_case", [
	{
		'name': 'single_action',
		'keywords': ['keyword1', 'keyword2'],
		'map': {
			'keyword1': {'action'},
			'keyword2': {'action'}
		},
		'expected': 'action'
	},
	{
		'name': 'multiple_actions',
		'keywords': ['keyword1', 'keyword2', 'keyword3'],
		'map': {
			'keyword1': {'action1'},
			'keyword2': {'action1'},
			'keyword3': {'action2'}
		},
		'expected': 'action1'
	},
	{
		'name': 'less_frequency',
		'keywords': ['keyword1', 'keyword2', 'keyword3'],
		'map': {
			'keyword1': {'action1'},
			'keyword2': {'action2'},
			'keyword3': {'action3'}
		},
		'expected': None
	},
	{
		'name': 'single_keyword_multiple_actions',
		'keywords': ['keyword1', 'keyword2'],
		'map': {
			'keyword1': {'action1', 'action2'},
			'keyword2': {'action2'}
		},
		'expected': 'action2'
	}
])
def test_predict_action_frequency(test_case):
	service = make_mock_service()
	service._wrapper.get_actions_for_keyword = lambda action_keyword: test_case['map'][action_keyword]

	result = service.predict_action_frequency(test_case['keywords'], 0.5)

	assert result == test_case['expected']

@pytest.mark.parametrize("test_case", [
	{
		'name': 'high_probability',
		'actions': [('action1', 0.7), ('action2', 0.3)],
		'expected': 'action1'
	},
	{
		'name': 'user_choice',
		'actions': [('action1', 0.4), ('action2', 0.4), ('action3', 0.2)],
		'input': '1',
		'expected': 'action1'
	},
	{
		'name': 'skip_request',
		'actions': [('action1', 0.4), ('action2', 0.4), ('action3', 0.2)],
		'input': '4',
		'expected': None
	},
	{
		'name': 'invalid_input',
		'actions': [('action1', 0.4), ('action2', 0.4), ('action3', 0.2)],
		'input': '99',
		'expected': None
	}
])
def test_predict_action_classification(monkeypatch, test_case):
	service = make_mock_service()
	service._wrapper.predict_top_actions = lambda action_keywords, max_possibilities, probability_cutoff: test_case['actions']
	service._wrapper.get_action_description = MagicMock()
	monkeypatch.setattr("builtins.input", lambda _: test_case["input"])
	service._wrapper.train_action_pipeline = MagicMock()

	result = service.predict_action_classification([], 5, 0.5)

	assert result == test_case['expected']

@pytest.mark.parametrize("test_case", [
	{
		'name': 'single_argument',
		'keywords': ['keyword1', 'keyword2'],
		'map': {
			'keyword1': {'argument'},
			'keyword2': {'argument'}
		},
		'expected': 'argument'
	},
	{
		'name': 'multiple_arguments',
		'keywords': ['keyword1', 'keyword2', 'keyword3'],
		'map': {
			'keyword1': {'argument1'},
			'keyword2': {'argument1'},
			'keyword3': {'argument2'}
		},
		'expected': 'argument1'
	},
	{
		'name': 'less_frequency',
		'keywords': ['keyword1', 'keyword2', 'keyword3'],
		'map': {
			'keyword1': {'argument1'},
			'keyword2': {'argument2'},
			'keyword3': {'argument3'}
		},
		'expected': None
	},
	{
		'name': 'single_keyword_multiple_arguments',
		'keywords': ['keyword1', 'keyword2'],
		'map': {
			'keyword1': {'argument1', 'argument2'},
			'keyword2': {'argument2'}
		},
		'expected': 'argument2'
	}
])
def test_predict_argument_frequency(test_case):
	service = make_mock_service()
	service._wrapper.get_argument_indices_for_keyword = lambda action, argument_keyword: test_case['map'][argument_keyword]

	result = service.predict_argument_frequency('action', test_case['keywords'], 0.5)

	assert result == test_case['expected']

@pytest.mark.parametrize("test_case", [
	{
		'name': 'high_probability',
		'arguments': [(0, 0.7), (1, 0.3)],
		'argument_group': ([], {('nk1', False), ('nk2', False)}),
		'expected': [(0, None)]
	},
	{
		'name': 'user_choice_priority',
		'arguments': [(0, 0.4), (1, 0.4), (2, 0.2)],
		'argument_group': ([], {('nk1', True), ('nk2', False)}),
		'input': '1',
		'expected': [(0, 'nk1')]
	},
	{
		'name': 'user_choice_nonpriority',
		'arguments': [(0, 0.4), (1, 0.4), (2, 0.2)],
		'argument_group': ([], {('nk1', False), ('nk2', False)}),
		'input': '1',
		'expected': [(0, 'nk1'), (0, 'nk2')]
	},
	{
		'name': 'skip_request',
		'arguments': [(0, 0.4), (1, 0.4), (2, 0.2)],
		'argument_group': ([], {('nk1', False), ('nk2', False)}),
		'input': '7',
		'expected': [(None, None)]
	},
	{
		'name': 'invalid_input',
		'arguments': [(0, 0.4), (1, 0.4), (2, 0.2)],
		'argument_group': ([], {('nk1', False), ('nk2', False)}),
		'input': '99',
		'expected': [(None, None)]
	}
])
def test_predict_argument_nonkeyword_classification(monkeypatch, test_case):
	service = make_mock_service()
	service._wrapper.predict_top_arguments_indices = lambda action, keywords, max_possibilities, probability_cutoff: test_case['arguments']
	service._wrapper.get_argument_description = MagicMock()
	service._wrapper.get_argument_type = MagicMock(return_value = 'any')
	monkeypatch.setattr("builtins.input", lambda _: test_case["input"])
	service._wrapper.train_argument_pipeline = MagicMock()

	result = service.predict_argument_nonkeyword_classification('', test_case['argument_group'], 10, 0.5)

	assert any(result == expected for expected in test_case['expected'])

@pytest.mark.parametrize("test_case", [
	{
		'name': 'whitespace_seperated',
		'query': 'word1 word2',
		'expected': [('word1', False), ('word2', False)],
		'raises': False
	},
	{
		'name': 'quoted',
		'query': 'word1 "word2" \'word3\'',
		'expected': [('word1', False), ('word2', True), ('word3', True)],
		'raises': False
	},
	{
		'name': 'invalid_quotes',
		'query': 'word1 "word2 word3',
		'raises': True
	},
	{
		'name': 'empty_quotes_ignore',
		'query': 'word1 ""',
		'expected': [('word1', False)],
		'raises': False
	}
])
def test_extract_tokens(test_case):
	service = make_mock_service()

	if test_case['raises']:
		with pytest.raises(Exception):
			service.extract_tokens(test_case['query'])
	else:
		result = service.extract_tokens(test_case['query'])

		assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'keyword_flush',
		'query': 'word1 word2 word3',
		'keywords': {'word2'},
		'expected': (['word2'], [[('word1', False)], [('word3', False)]])
	},
	{
		'name': 'keyword_quoted',
		'query': 'word1 "word2" word3',
		'keywords': {'word2'},
		'expected': ([], [[('word1', False), ('word2', True), ('word3', False)]])
	}
])
def test_extract_action_groups(test_case):
	service = make_mock_service()
	service._wrapper.match_action_keyword = lambda token, probability_cutoff: token if token in test_case['keywords'] else None

	tokens = service.extract_tokens(test_case['query'])
	result = service.extract_action_groups(tokens)

	assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'nkkeyword_flush',
		'action_groups': [[('word1', False), ('word2', False), ('word3', False)]],
		'keywords': {'word2'},
		'stop_words': set(),
		'expected': ([('word2', {('word3', False)})], [('word1', False)])
	},
	{
		'name': 'nk_quoted',
		'action_groups': [[('word1', False), ('word2', False), ('word3', True)]],
		'keywords': {'word2', 'word3'},
		'stop_words': set(),
		'expected': ([('word2', {('word3', True)})], [('word1', False)])
	},
	{
		'name': 'nk_stopword',
		'action_groups': [[('word1', False), ('word2', False), ('word3', False)]],
		'keywords': {'word2'},
		'stop_words': {'word1'},
		'expected': ([('word2', {('word3', False)})], [])
	},
	{
		'name': 'multiple_action_groups',
		'action_groups': [[('word1', False), ('word2', False), ('word3', False), ('word4', False)], [('word5', False), ('word6', False)]],
		'keywords': {'word1', 'word3', 'word5'},
		'stop_words': set(),
		'expected': ([('word1', {('word2', False)}), ('word3', {('word4', False)}), ('word5', {('word6', False)})], [])
	},
	{
		'name': 'empty_nonkeyword_flush',
		'action_groups': [[('word1', False), ('word2', False)]],
		'keywords': {'word2'},
		'stop_words': set(),
		'expected': ([], [('word1', False)])
	},
])
def test_extract_argument_groups(test_case):
	service = make_mock_service()
	service._wrapper.match_argument_keyword = lambda action, token, probability_cutoff: token if token in test_case['keywords'] else None
	service._wrapper.is_stop_word = lambda token, probability_cutoff: token in test_case['stop_words']

	result = service.extract_argument_groups('action', test_case['action_groups'])

	assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'types',
		'non_keywords': [('string', False), ('123', False), ('string123', False)],
		'expected': ({
			'str': ['string'],
			'int': ['123'],
			'any': ['string123']
		}, {})
	},
	{
		'name': 'quoted',
		'non_keywords': [('string', False), ('123', True), ('string123', False)],
		'expected': ({
			'str': ['string'],
			'any': ['string123']
		}, {
			'int': ['123']
		})
	}
])
def test_extract_classified_nonkeywords(test_case):
	service = make_mock_service()

	result = service.extract_classified_nonkeywords(test_case['non_keywords'])

	assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'assigned unassigned',
		'arguments': ['arg1', None, 'arg2'],
		'required_arguments': [0, 1, 2],
		'optional_arguments': [],
		'expected': (
			[(0, 'arg1'), (2, 'arg2')],
			[],
			[1],
			[]
		)
	},
	{
		'name': 'required optional',
		'arguments': ['arg1', None, 'arg2', None],
		'required_arguments': [0, 1],
		'optional_arguments': [2, 3],
		'expected': (
			[(0, 'arg1')],
			[(2, 'arg2')],
			[1],
			[3]
		)
	}
])
def test_extract_argument_indices_information(test_case):
	service = make_mock_service()
	service._wrapper.get_required_arguments = lambda action: test_case['required_arguments']
	service._wrapper.get_optional_arguments = lambda action: test_case['optional_arguments']

	result = service.extract_argument_indices_information('action', test_case['arguments'])

	assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'all_type_match',
		'indices': [0, 1, 2],
		'types': ['str', 'int', 'any'],
		'non_keywords': [('string', False), ('123', False), ('string123', False)],
		'input': '99',
		'expected': [(0, 'string'), (1, '123'), (2, 'string123')]
	},
	{
		'name': 'borrow_to_any',
		'indices': [0, 1, 2],
		'types': ['str', 'int', 'any'],
		'non_keywords': [('string', False), ('123', False), ('strings', False)],
		'input': '1',
		'expected': [(0, 'string'), (1, '123'), (2, 'strings')]
	},
	{
		'name': 'not_enough',
		'indices': [0, 1, 2],
		'types': ['str', 'int', 'any'],
		'non_keywords': [('string', False), ('string123', False)],
		'input': '99',
		'expected': [(0, 'string'), (1, None), (2, 'string123')]
	},
	{
		'name': 'priority',
		'indices': [0, 1],
		'types': ['str', 'int'],
		'non_keywords': [('string', True), ('strings', False), ('123', False), ('456', True)],
		'input': '99',
		'expected': [(0, 'string'), (1, '456')]
	}
])
def test_extract_arguments_questions_nonkeywords(monkeypatch, test_case):
	service = make_mock_service()
	service._wrapper.get_argument_type = lambda action, idx: test_case['types'][idx]
	service._wrapper.get_argument_description = lambda action, idx: 'description'
	monkeypatch.setattr("builtins.input", lambda _: test_case["input"])

	result = service.extract_arguments_questions_nonkeywords('action', test_case['indices'], test_case['non_keywords'])

	assert result == test_case['expected']

@pytest.mark.parametrize('test_case', [
	{
		'name': 'first_check_existing_app',
		'existing_app': 1,
		'monitored_app': 0,
		'match_nickname': 0,
		'get_nickname': 0,
		'match_class': 0,
		'get_class': 0
	},
	{
		'name': 'second_check_monitored_app',
		'existing_app': 0,
		'monitored_app': 1,
		'match_nickname': 0,
		'get_nickname': 0,
		'match_class': 0,
		'get_class': 0
	},
	{
		'name': 'third_check_nickname',
		'existing_app': 0,
		'monitored_app': 0,
		'match_nickname': 1,
		'get_nickname': 1,
		'match_class': 0,
		'get_class': 0
	},
	{
		'name': 'fourth_check_class',
		'existing_app': 0,
		'monitored_app': 0,
		'match_nickname': 0,
		'get_nickname': 0,
		'match_class': 1,
		'get_class': 1
	}
])
def test_extract_app(test_case):
	service = make_mock_service()
	service._wrapper.match_existing_app = MagicMock(return_value = 'app_name' if test_case['existing_app'] else None)
	service._wrapper.match_monitored_app = MagicMock(return_value = 'app_name' if test_case['monitored_app'] else None)
	service._wrapper.match_nickname = MagicMock(return_value = 'nickname' if test_case['match_nickname'] else None)
	service._wrapper.get_app_for_nickname = MagicMock(return_value = 'app_name' if test_case['get_nickname'] else None)
	service._wrapper.match_class = MagicMock(return_value = 'class' if test_case['match_class'] else None)
	service._wrapper.get_mostused_app_for_class = MagicMock(return_value = 'app_name' if test_case['get_class'] else None)

	service.extract_app('token')

	assert service._wrapper.match_existing_app.call_count == test_case['existing_app'] if test_case['existing_app'] else True
	assert service._wrapper.match_monitored_app.call_count == test_case['monitored_app'] if test_case['monitored_app'] else True
	assert service._wrapper.match_nickname.call_count == test_case['match_nickname'] if test_case['match_nickname'] else True
	assert service._wrapper.get_app_for_nickname.call_count == test_case['get_nickname'] if test_case['get_nickname'] else True
	assert service._wrapper.match_class.call_count == test_case['match_class'] if test_case['match_class'] else True
	assert service._wrapper.get_mostused_app_for_class.call_count == test_case['get_class'] if test_case['get_class'] else True