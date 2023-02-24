from enum import Enum, auto
import unittest

def test_convert():
	assertEqual(DeviceType.convert("cuda"), DeviceType.CUDA)
		
