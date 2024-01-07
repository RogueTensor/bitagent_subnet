<div align="center">

# **BitQnA Subnet (#20) on Bittensor** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1175085112703078400)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Q&A with Your Data <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Running](#Running)
- [License](#license)

## Introduction

The BitQnA project is driving to provide an advanced AI-driven Q&A platform that seamlessly integrates with diverse data sources, from text documents to multimedia, providing deep, contextually relevant answers in real-time. This system, independent of major AI APIs, is tailored for any downstream application, from legal research to IT support, and includes advanced features such as cascading summarization and Chain of Code reasoning.

### Key Features

- Q&A with RAG, Cascading Summarization (map reduce) and Chain of Code
- User provides their own data
- No API requirements
- Powers discord bots and web applications
  
---

## Running

- You can run this subnet either as a subnet owner, as a subnet validator or as a subnet miner. 
- **IMPORTANT:** Make sure you are aware of the minimum compute requirements for your subnet. See the [Minimum compute YAML configuration](./min_compute.yml).

### Install

- **Running locally**: Follow the step-by-step instructions described in this section: [Running Subnet Locally](./docs/running_on_staging.md).
- **Running on Bittensor testnet**: Follow the step-by-step instructions described in this section: [Running on the Test Network](./docs/running_on_testnet.md).
- **Running on Bittensor mainnet**: Follow the step-by-step instructions described in this section: [Running on the Main Network](./docs/running_on_mainnet.md).

### Simple Setup
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/RogueTensor/bitqna_subnet
cd bitqna_subnet
python -m pip install -r requirements.txt
python -m pip install -e .
```

After getting the [subtensor package started and a subnet up and running](./docs/running_on_staging.md) (for staging/local) - you can use this [script](./scripts/setup_and_run.sh) to:
- create wallets (for owner, validators, miners),
- fund those wallets with the right amount of tao,
- register wallets on the local subnet,
- start miners and validators

```bash
./script/setup_and_run.sh
```
You can use several flags to configure:
- the number of miners or validators it sets up,
- whether it funds wallets,
- or if it registers wallets,
- or just launches a miner
```bash
$ ./scripts/setup_and_run.sh -h

Creates wallets for the subnet (owner, validators, miners), funds them, registers them, then starts them.

usage: ./scripts/setup_and_run.sh --num_validators num --num_miners num --subnet_prefix string

  --num_validators num     number of validators to launch
                           (default: 1)
  --num_miners     num     number of miners to launch
                           (default: 2)
  --subnet_prefix  string  the prefix of the subnet wallets
                           (default: local_subnet_testing_bitqna)
  --skip-wallet    num     pass in 0 to skip wallet creation
                           (default: 1)
  --skip-faucet    num     pass in 0 to skip wallet funding
                           (default: 1)
  --skip-subnet    num     pass in 0 to skip subnet creation
                           (default: 1)
  --skip-reg       num     pass in 0 to skip all registration to the subnet
                           (default: 1)
  --skip-val-reg   num     pass in 0 to skip validator registration to the subnet
                           (default: 1)
  --skip-miner-reg num     pass in 0 to skip miner registration to the subnet
                           (default: 1)
  --skip-launch    num     pass in 0 to skip validator and miner launching on the subnet
                           (default: 1)
  --skip-launch_v  num     pass in 0 to skip validator on the subnet
                           (default: 1)
  --only-launch    num     pass in 0 to skip everything but launching
                           (default: 1)

Example: ./scripts/setup_and_run.sh --only-launch
This will skip everything and just launch the already registered and funded validators and miners
```

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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
```
