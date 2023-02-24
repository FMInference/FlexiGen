from enum import Enum, auto
from flexgen.utils import ValueHolder

def test_value():
    value = ValueHolder(50)
    value.store(50)
    assert wallet.val == 50
		
