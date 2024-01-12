<div align="center">

# **BitAgent Subnet (#20) on Bittensor** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1175085112703078400)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Tasking and Q&A with Your Data and Your World <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Get Running](#get-running)
  - [BitAgent](#bitagent)
  - [Validator](#validator)
  - [Miner](#miner)
    - [Miner Feedback](#miner-feedback)
  - [Advanced](#advanced)
- [FAQ](#faq)
- [License](#license)

## Introduction

BitAgent has 2 core thrusts:

1) **Q&A/Tasking** - comes in a few flavors - a) with your data in real time (BYOD), b) summarization of large data (BYOD), c) logic-based reasoning and d) agency (tool execution, operation performance), 
examples: 
     - Fill in this form, from this source data, to match the tone and professionalism of these prior examples
     - Plot the occurrence of key topics as they accumulate in this data
     - Provide metrics to align with these requirements and provide a test suite in python
     - Grab the last 3 weeks of publications from arxiv that have to do with Generative AI and provide a summary for each
     - We just received this support ticket, here's our knowledge base, please update the ticket with a procedure for the tier 1 support to follow

2) **Integrated Orchestration** - this is task completion initiated by natural language for application, 
examples:
    - You're using the browser plugin/extension built on this subnet (Coming Soon) - you can Q&A from that website about that website
    - Again with the subnet's browser plugin - you're on a really complex web page and you just can't concentrate, it's too intense - you head over to the browser plugin and you click on the provided ELI5 (explain like I'm 5) button to convert all the complex text on the page to easy-to-understand text.
    - Again, with the plugin, you're on amazon and you're met with a TON of reviews, most of which look fake - your task to the subnet (via natural language to the plugin) is to "Hide all the reviews on this page that appear fake."
    - Again, with our browser plugin, you're in YouTube and this video isn't getting to the point, so you task the subnet to "Skip ahead to the second instance where they begin talking about whatever"
    - This time you own your own company, let's say you own an IT support company that works with legal firms, dentist offices, etc.  You've collected a knowledge base over many years and you never know what requests will come through.  A few seconds ago, a new, verified request came in from the head of HR at some legal firm, letting you know that <so-in-so> just joined the team and needs their accounts and access setup.  Using our subnet API, you can have the subnet be your first line of defense, by doing the tasks that you give it access to perform. 

To be successful, Thrust 2 requires all aspects of Thrust 1, so we're working initial efforts in those areas.
However our future vision is to leverage and integrate other subnets for some/all of Thrust 1 and provide SOTA (state-of-the-art) capabilities in the other areas.

### Key Features

- Downstream applications towards this vision are in the works:
  - Discord bot - to, at a minimum, provide an answer to the seemingly daily (hourly?) question about getting testnet tao ;)
  - Web and Form Filler applications
  - Browser plugin (see examples above)
- Leveraging concepts from [voyager](https://voyager.minedojo.org/) / [blenderGPT](https://github.com/gd3kr/BlenderGPT) / [chain of code](https://chain-of-code.github.io/) for real world applications
- BYOD and real-time ingestion / operation
- Very low barrier-to-entry out the gate
  - low miner / validator requirements
  - no GPU requirement
  - No API / subscription requirements
  - and a provided miner that solves most of the current challenges/tasks
- Miner's receive [transparent feedback](#miner-feedback)
- And a BONUS for getting this far - are you tired of waiting for registration slots?  Check out [register.sh](./scripts/register.sh)
  
---

## Get Running

- Before getting too far, please make sure you've looked over the [Bittensor documentation](https://docs.bittensor.com/) for you needs.
- For now, the min compute requirements are REALLY low - the validators and miners currently use an LLM that can run on CPU, with a very small footprint (800M params).

### BitAgent
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/RogueTensor/bitagent_subnet
cd bitagent_subnet
python -m pip install -r requirements.txt
python -m pip install -e .
```

Then make sure to register your intended wallet (coldkey, hotkey) to subnet 20:
```bash
btcli subnet register --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network finney --netuid 20
```

### Validator

Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```

If you just want to run the validator without the [script](./scripts/setup_and_run.sh) or are connecting to mainnet:
```bash
# for testing
python3 neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>

# for mainnet
pm2 start neurons/validator.py --interpreter python3 -- --netuid 20 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --axon.port <PORT>

# for mainnet with AUTO UPDATES (recommended)
pm2 start run.sh --name bitagent_validators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key> --netuid 20
```

### Miner
If you just want to run the miner without the [script](./scripts/setup_and_run.sh) or are connecting to mainnet:
```bash
# for testing
python3 neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>
# for mainnet
pm2 start neurons/miner.py --interpreter python3 -- --netuid 20 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --axon.port <PORT>
```

#### Miner Feedback
As a miner, you receive tasks, you get rewarded, but often you do not know what you're being graded on.
BitAgent offers transparent feedback (in debug mode), so you know what you're up against.

Here's an example of a well performed task:
![miner feedback - good example](./docs/examples/output_to_miner.png)

Here's an example of a poorly performed task:
![miner feedback - bad example](./docs/examples/bad_output_to_miner.png)

### Advanced
If you have a need to create and fund wallets for your own testing ...

After getting the [subtensor package started and a subnet up and running](./docs/running_on_staging.md) (for staging/local) - you can use this [script](./scripts/setup_and_run.sh) to:
- create wallets (for owner, validators, miners),
- fund those wallets with the right amount of tao,
- register wallets on the local subnet,
- start miners and validators

```bash
./scripts/setup_and_run.sh
```
You can use several flags to configure:
- the number of miners or validators it sets up,
- whether it funds wallets,
- or if it registers wallets,
- or just launches a miner
```bash
bitagent_subnet$ ./scripts/setup_and_run.sh --help

Creates wallets for the subnet (owner, validators, miners), funds them, registers them, then starts them.

usage: ./scripts/setup_and_run.sh --num_validators num --num_miners num --subnet_prefix string

  --num_validators num     number of validators to launch
                           (default: 1)
  --num_miners     num     number of miners to launch
                           (default: 2)
  --subnet_prefix  string  the prefix of the subnet wallets
                           (default: local_subnet_testing_bitagent)
  --skip-wallet            skip wallet creation
                           (default: run wallet creation)
  --skip-faucet            skip wallet funding
                           (default: fund wallets)
  --skip-subnet            skip subnet creation
                           (default: create subnet)
  --skip-reg               skip all registration to the subnet
                           (default: register wallets)
  --skip-val-reg           skip validator registration to the subnet
                           (default: register validator)
  --skip-miner-reg         skip miner registration to the subnet
                           (default: register miner)
  --skip-launch            skip validator and miner launching on the subnet
                           (default: launch validators and miners)
  --skip-launch_v          skip validator launching on the subnet
                           (default: launch validators)
  --only-launch            skip everything but launching
                           (default: do everything)
  --test-net               do the same things, but for testnet
                           (default: false, local)
  --main-net               do the same things, but for mainnet
                           (default: false, local)
  --netuid                 the netuid to work with
                           (default: 1 for local, change if main or test)

Example: ./scripts/setup_and_run.sh --only-launch
This will skip everything and just launch the already registered and funded validators and miners
```
---

## FAQ
Q: How much GPU (VRAM/cores) do I need to run a miner in Phase 1?
A: You don't need a GPU at all, the LLMs being used can run on CPU.

Q: I am seeing: RuntimeWarning: coroutine 'Server.serve' was never awaited - help?
A: Asked and [answered by Carro](https://discord.com/channels/799672011265015819/1161765008347254915/1188957429392756818)

Q: Are there any required subscriptions or APIs?
A: No - no subs, no external companies, in fact we'd rather the community build something amazing rather than relying on corporations.

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
