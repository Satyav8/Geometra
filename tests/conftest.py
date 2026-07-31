import os
import sys
import tempfile

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["SQLITE_DB_PATH"] = _test_db_path

from database import init_db  # noqa: E402

init_db()
