import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.cleaning import remove_think_tags

def test_add():
    text = "This is some text <think>remove this part</think> and keep this."
    output = remove_think_tags(text)

    assert output == "This is some text  and keep this."