from enum import Enum, auto
from flexgen.flex_opt import TorchDevice
from flexgen.flex_opt import DeviceType
import unittest

class MyTestCase(unittest.TestCase):
	def test_convert(self):
		self.assertEqual(DeviceType.convert("cuda"), DeviceType.CUDA)
