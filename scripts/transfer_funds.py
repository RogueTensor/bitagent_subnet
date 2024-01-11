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

parser = argparse.ArgumentParser(description='transfer amount to dest wallet, given hotkey and coldkey names')
parser.add_argument('--hotkey_name', type=str, required=True)
parser.add_argument('--coldkey_name', type=str, required=True)
parser.add_argument('--dest', type=str, required=True)
parser.add_argument('--amount', type=float, required=True)
parser.add_argument('--network', type=str, required=False, default="test")

args = parser.parse_args()
print(args)

# Bittensor's chain interface.
subtensor = bt.subtensor(network=args.network) 
subtensor.get_current_block()

# wallet
wallet = bt.wallet(name=args.coldkey_name, hotkey=args.hotkey_name)

# Transfer Tao to a destination address.
subtensor.transfer(wallet=wallet, dest=args.dest, amount=args.amount)
