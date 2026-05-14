import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from requests import (
    check_file_sufficiency,
    check_function_sufficiency,
)


def test_check_file_sufficiency_complete():
    """Test that complete file data passes sufficiency check."""
    
    data = {
        "type": "file_content",
        "path": "/tmp/test.py",
        "content": {
            1: "# Full file",
            2: "def target_func():",
            3: "    return 42",
            4: ""
        },
        "EOF": True,
        "line_count": 4,
        "source_info": {
            "call_count": 1,
            "total_bytes_read": 100
        }
    }
    
    pending_info = {
        "path": "/tmp/test.py",
        "function_name": "target_func",
        "full_code": "    return 43\n"
    }
    
    assert check_file_sufficiency(data, pending_info) is not None


def test_check_file_sufficiency_incomplete():
    """Test that incomplete file data fails sufficiency check."""
    
    data = {
        "type": "file_content",
        "path": "/tmp/test.py",
        "content": {
            "1": "# Partial file"
        },
        "EOF": False
    }
    
    pending_info = {
        "path": "/tmp/test.py",
        "function_name": "target_func",
        "full_code": "    return 43\n"
    }
    
    assert check_file_sufficiency(data, pending_info) is None


def test_check_function_sufficiency_complete():
    """Test that data with complete function passes sufficiency check."""
    
    data = {
        "type": "file_content",
        "path": "/tmp/test.py",
        "content": {
            1: "# Some module",
            2: "",
            3: "def helper():",
            4: "    pass",
            5: "",
            6: "def target_func(x, y):",
            7: "    return x + y",
            8: "",
            9: "def other():",
            10: "    pass"
        },
        "EOF": True,
        "line_count": 10
    }
    
    pending_info = {
        "path": "/tmp/test.py",
        "function_name": "target_func",
        "full_code": "    return x * y\n"
    }
    
    assert check_function_sufficiency(data, "target_func") is not None


def test_check_function_sufficiency_incomplete():
    """Test that data without the target function fails sufficiency check."""
    
    data = {
        "type": "file_content",
        "path": "/tmp/test.py",
        "content": {
            1: "# Some module",
            2: "",
            3: "def helper():",
            4: "    pass"
        },
        "EOF": True,
        "line_count": 4
    }
    
    pending_info = {
        "path": "/tmp/test.py",
        "function_name": "target_func",
        "full_code": "    return x * y\n"
    }
    
    assert check_function_sufficiency(data, "target_func") is None


def test_check_file_sufficiency_no_content():
    """Test that data with no content fails."""
    
    data = {
        "type": "file_content",
        "path": "/tmp/test.py",
        "content": None
    }
    
    pending_info = {
        "path": "/tmp/test.py",
        "function_name": "target_func",
        "full_code": "    return 43\n"
    }
    
    assert check_file_sufficiency(data, pending_info) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])