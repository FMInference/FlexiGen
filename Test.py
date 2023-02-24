from enum import Enum, auto
from flexgen.utils import ValueHolder

def test_value():
    retrn = ValueHolder()
    retrn.pop()
    assert retrn.ret == retrn.val
