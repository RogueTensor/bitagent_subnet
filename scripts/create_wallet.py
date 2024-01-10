# The MIT License (MIT)
# Copyright © 2023 RogueTensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import bittensor as bt

parser = argparse.ArgumentParser(description='create wallet, given hotkey and coldkey names')
parser.add_argument('--hotkey_name', type=str, required=True)
parser.add_argument('--coldkey_name', type=str, required=True)
parser.add_argument('--num', type=int, required=False, default=1)
parser.add_argument('--local', action=argparse.BooleanOptionalAction)

args = parser.parse_args()

for i in range(args.num):
  wallet = bt.wallet(name=f"{args.coldkey_name}_{i}", hotkey=f"{args.hotkey_name}_{i}")
  if args.local:
    print("#############################################")
    print("WARNING: Not going to use passwords for the coldkey")
    print("Pass --local False, to require passwords")
    print("#############################################")
    wallet.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
  else:
    wallet.create_if_non_existent() # defaults to use password for coldkey
  print(f"Created (if needed) wallet for coldkey: {args.coldkey_name}_{i}")
