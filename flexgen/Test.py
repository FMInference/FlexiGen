from pytorch_backend import DeviceType
from enum import Enum, auto

def test_convert():
	assertEqual(DeviceType.convert("cuda"), DeviceType.CUDA)
		
