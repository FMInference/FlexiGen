## Example commands

Example commands for OPT-30B and OPT-66B on machines with 32GB of system RAM and 24 GB of VRAM.
```
python completion.py --model facebook/opt-30b --percent 100 0 100 0 100 0 --compress-weight
python completion.py --model facebook/opt-66b --percent 50 10 100 0 100 0 --compress-weight
```
