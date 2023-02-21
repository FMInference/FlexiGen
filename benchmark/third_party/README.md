## Third party
Some benchmarks require the following third-party libraries.
We maintain forks of these librares under this repo. They can be installed as follows. 

### Transformers
huggingface/transformers: v4.24.0

```bash
cd FlexGen/benchmark/third_party/transformers
pip3 install -e .
pip3 install accelerate==0.15.0
```

### DeepSpeed
microsoft/DeepSpeed: v0.7.7

```bash
cd FlexGen/benchmark/third_party/DeepSpeed
pip3 install -e .

# (Optional) build asyncio ops
# sudo apt install libaio-dev gcc-multilib
# DS_BUILD_AIO=1 pip3 install -e .
```

### PagecacheManagement
```bash
cd FlexGen/third_party/pagecache-mangagement/trunk
make
sudo cp *.so /usr/local/lib/
sudo cp *.sh /usr/local/bin/
```
