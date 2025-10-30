from unittest.mock import patch, MagicMock, mock_open
import threading
from pathlib import Path

from Include.wrapper.sqlite_wrapper import SQLiteWrapper

@patch('sqlite3.connect')
def test_initialize_db(mock_connect):
    mock_conn = MagicMock()

    mock_connect.return_value = mock_conn
    db = SQLiteWrapper(':memory:')
    
    # Check if PRAGMA statements were executed
    mock_conn.execute.assert_any_call("PRAGMA journal_mode=WAL;")
    mock_conn.execute.assert_any_call("PRAGMA synchronous=NORMAL;")
    mock_conn.execute.assert_any_call("PRAGMA foreign_keys=ON;")

@patch('sqlite3.connect')
def test_execute_single_query(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    db = SQLiteWrapper(':memory:')
    db.execute("INSERT INTO test (id) VALUES (?)", (1,))
    
    assert mock_conn.execute.call_args == (
        ("INSERT INTO test (id) VALUES (?)", (1,)),
        {}
    )

@patch('sqlite3.connect')
def test_execute_many_queries(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    db = SQLiteWrapper(':memory:')
    params = [(1,), (2,), (3,)]
    db.execute_many("INSERT INTO test (id) VALUES (?)", params)
    
    assert mock_conn.executemany.call_args == (
        ("INSERT INTO test (id) VALUES (?)", params),
        {}
    )

@patch('sqlite3.connect')
def test_fetch_all_results(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    expected_results = [{'id': 1}, {'id': 2}]
    mock_cursor.fetchall.return_value = expected_results
    
    db = SQLiteWrapper(':memory:')
    results = db.fetchall("SELECT * FROM test")
    
    assert results == expected_results

@patch('sqlite3.connect')
def test_fetch_one_result(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    expected_result = {'id': 1}
    mock_cursor.fetchone.return_value = expected_result
    
    db = SQLiteWrapper(':memory:')
    result = db.fetchone("SELECT * FROM test WHERE id = ?", (1,))
    
    assert result == expected_result

@patch('sqlite3.connect')
def test_transaction_commit(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    db = SQLiteWrapper(':memory:')
    with db.transaction() as tx:
        tx.execute("INSERT INTO test (id) VALUES (?)", (1,))
        tx.execute_many("INSERT INTO test (id) VALUES (?)", [(2,), (3,)])
    
    mock_conn.execute.assert_any_call("BEGIN")
    mock_conn.execute.assert_any_call("INSERT INTO test (id) VALUES (?)", (1,))
    mock_conn.executemany.assert_called_once_with("INSERT INTO test (id) VALUES (?)", [(2,), (3,)])
    mock_conn.commit.assert_called_once()

@patch('sqlite3.connect')
def test_transaction_rollback(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    db = SQLiteWrapper(':memory:')
    try:
        with db.transaction() as tx:
            tx.execute("INSERT INTO test (id) VALUES (?)", (1,))
            raise Exception("Test error")
    except Exception:
        pass
    
    mock_conn.execute.assert_any_call("BEGIN")
    mock_conn.execute.assert_any_call("INSERT INTO test (id) VALUES (?)", (1,))
    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()

@patch('builtins.open', new_callable=mock_open, read_data="CREATE TABLE test (id INTEGER PRIMARY KEY);")
@patch('sqlite3.connect')
def test_execute_script(mock_connect, mock_file):
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    
    db = SQLiteWrapper(':memory:')
    db.execute_script('test.sql')
    
    mock_file.assert_called_once_with('test.sql', 'r')
    mock_conn.executescript.assert_called_once_with("CREATE TABLE test (id INTEGER PRIMARY KEY);")

@patch('builtins.open', new_callable=mock_open, read_data="SELECT * FROM test;")
@patch('sqlite3.connect')
def test_fetch_script(mock_connect, mock_file):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_connect.return_value = mock_conn
    
    expected_results = [{'id': 1}, {'id': 2}]
    mock_cursor.fetchall.return_value = expected_results
    
    db = SQLiteWrapper(':memory:')
    results = db.fetch_script('test.sql')
    
    assert results == expected_results

def test_thread_safety():
    # Use a temporary file instead of :memory: to test thread safety
    import tempfile
    import os
    
    temp_db = tempfile.NamedTemporaryFile()
    temp_db.close()
    
    try:
        db = SQLiteWrapper(temp_db.name)
        results = []
        errors = []
        
        # Create table before threading to avoid race condition
        with db.transaction() as tx:
            tx.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
        
        def worker(worker_id):
            try:
                with db.transaction() as tx:
                    tx.execute("INSERT INTO test (id) VALUES (?)", (worker_id,))
                result = db.fetchone("SELECT id FROM test WHERE id = ?", (worker_id,))
                results.append(result['id'] if result else None)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert sorted(results) == list(range(5))
    finally:
        # Clean up the temporary file
        os.unlink(temp_db.name)

def test_connection_timeout():
    db = SQLiteWrapper(':memory:')
    
    # Verify that connection is created with timeout
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        with db._get_conn():
            pass
        
        mock_connect.assert_called_once_with(
            Path(':memory:'),
            timeout=30,
            isolation_level=None
        )