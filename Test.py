from enum import Enum, auto
from flexgen.pytorch_backend import DeviceType
import unittest

class MyTestCase(unittest.TestCase):
	def test_convert(self):
		self.assertEqual(DeviceType.convert("cuda"), DeviceType.CUDA)
