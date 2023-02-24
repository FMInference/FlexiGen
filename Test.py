from enum import Enum, auto
from flexgen.utils import ValueHolder

def test_value():
    value = ValueHolder()
    value.store(50)
    assert value.val == 50
		
