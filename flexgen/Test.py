from enum import Enum, auto
import unittest

class MyTestCase(unittest.TestCase):
	def test_convert(self):
		self.assertEqual(DeviceType.convert("cuda"), DeviceType.CUDA)
		
