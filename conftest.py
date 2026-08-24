"""pytest configuration: ensures project root is on sys.path."""
import sys
from pathlib import Path

# Add project root to sys.path so tests can import src.* directly
sys.path.insert(0, str(Path(__file__).parent))
