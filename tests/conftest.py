from pathlib import Path

def pytest_sessionfinish(session, exitstatus):
    # File extensions to delete
    extensions = ["*.h5", "*.chk", "*.xlsx"]

    # Directories to clean: current dir and `tests/`
    dirs = [Path("."), Path("tests")]

    for d in dirs:
        for pattern in extensions:
            for file in d.glob(pattern):
                try:
                    file.unlink()
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Failed to delete {file}: {e}")
