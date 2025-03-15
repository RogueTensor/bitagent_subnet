<div align="center">

# **BitAgent Subnet (#20) on Bittensor** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1175085112703078400)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Agency for Your World Through Natural Language <!-- omit in toc -->

**Communications:** [BitAgent Discord](https://discord.com/channels/799672011265015819/1194736998250975332)\
**Downstream Applications:** [GoGoAgent](https://gogoagent.ai/) • [MSP Tech](https://msptech.ai)
</div>

---
- [Introduction](#introduction)
- [Get Running](#get-running)
  - [BitAgent](#bitagent)
  - [Validator](#validator)
    - [Dependencies](#dependencies)
    - [PM2 Installation](#pm2-installation)
    - [sglang Setup for Validators](#sglang-setup-for-validators)
    - [Recommended Startup](#recommended-startup)
    - [Alternative Startup](#alternative-startup)
    - [Verify Validator is Working](#verify-validator-is-working)
    - [Hardware Requirements](#validator-hardware-requirements)
  - [Miner](#miner)
    - [Hardware Requirements](#miner-hardware-requirements)
    - [Default Miner](#default-miner)
    - [Miner Emissions](#miner-emissions)
    - [Miner Considerations](#miner-considerations)
    - [Example Task](#example-task)
    - [Miner Feedback](#miner-feedback)
  - [Advanced](#advanced)
- [FAQ](#faq)
- [License](#license)

## Introduction

**Quick Pitch**: BitAgent revolutionizes how you manage tasks and workflows across platforms, merging the capabilities of large language models (LLMs) with the convenience of your favorite apps such as web browsers, Discord, and custom integrations, including other Bittensor subnets. BitAgent empowers users to seamlessly integrate intelligent agents, providing personalized assistance and integrated task automation.

**Key Objective** - provide intelligent agency to simplify and automate tasks in your day-to-day

**GoGoAgent - Our Application** - [https://gogoagent.ai](https://gogoagent.ai) \
**MSPTech - Real world business case** - [https://MSPTech.ai](https://msptech.ai)

**Key Features**
- Working our way up the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) (BFCL)
- No API / subscription requirements
- Run light models (8B parameter) for huge impact
- FINETUNED MODEL evaluation of tool calling language model fine tunes
- And a BONUS for getting this far - are you tired of waiting for registration slots?  Check out [register.sh](./scripts/register.sh)

---

## Get Running

- BitAgent is a competitive subnet, meaning miners succeed and fail based on how well they perform on tasks.
- **Make sure to test your miner locally before ever considering registering for Subnet 20.**
- Newly registered miners will have their model evaluated by every validator first before any incentive is received - this can take many hours.
- Before getting too far, please make sure you've looked over the [Bittensor documentation](https://docs.bittensor.com/) for your needs.
- The min compute requirements are [noted below for Validators](#validator-hardware-requirements).
- See [FAQ](#faq) for a few more details related to computing requirements for validators and miners.
- The minimum requirements for a miner are determined by the resources needed to train/fine-tune a competitive and performant tool calling LLM.

### BitAgent
This repository requires python 3.10 or higher. 
To install and get running, simply clone this repository and install the requirements.
```bash
git clone https://github.com/RogueTensor/bitagent_subnet
cd bitagent_subnet
# at this point, it's recommended that you use a venv, but not required; the next two lines are venv specific
python -m venv .venv #replace .venv with the name you'd like to use for your primary venv
source ./.venv/bin/activate
python -m pip install -e .
```

Then make sure to register your intended wallet (coldkey, hotkey) to Subnet 20:
```bash
btcli subnet register --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network finney --netuid 20
```

### Validator

#### Dependencies

You must have the following things:

- System with at least 48GB of VRAM
- Python >=3.10
- PM2

#### PM2 Installation

Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.\
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```

#### sglang Setup for Validators

You'll need to create a virtual env and install the requirements for sglang:
```bash
python3 -m venv .venvsglang
# note to change cu121 in this path according to this page: https://docs.flashinfer.ai/installation.html
./.venvsglang/bin/pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 
./.venvsglang/bin/pip install -r requirements.sglang.txt
```

**Test that it's working with:**
```
.venvsglang/bin/python -m sglang.launch_server --model-path Salesforce/xLAM-7b-r --port 8028 --host 0.0.0.0 --mem-fraction-static 0.40
```

You should not run out of memory and it should eventually show that the Salesforce model loaded correclty.

#### Recommended Startup

First, make sure you do the [sglang setup](#sglang-setup-for-validators) above.

```bash
# for mainnet with AUTO UPDATES (recommended)
pm2 start run.sh --name bitagent_validators_autoupdate -- --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key> --netuid 20
```

Double check everything is working by following [these steps](#verify-validator-is-working).

#### Alternative Startup

First, make sure you do the [sglang setup](#sglang-setup-for-validators) above.

```bash
# for testnet
python3 neurons/validator.py --netuid 76 --subtensor.network test --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>

# for mainnet
pm2 start neurons/validator.py --interpreter python3 -- --netuid 20 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --axon.port <PORT>
```

Double check everything is working by following [these steps](#verify-validator-is-working).

#### Verify Validator is Working

After you've launched and pm2 is running, here's what to expect:\
- You'll want to make sure miner's HF models are being evaluated - check your `pm2 log <ID> | grep OFFLINE` output for lines like these (from testnet):\
  ```bash
  1|bitagent | 2024-11-17 23:26:07.154 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Starting offline mode for competition 1-1
  1|bitagent | 2024-11-17 23:26:08.831 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Starting offline task
  1|bitagent | 2024-11-17 23:26:12.529 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Miner HF model names: [None, 'Salesforce/xLAM-7b-r']
  1|bitagent | 2024-11-17 23:26:12.529 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Unique miner HF model names: ['Salesforce/xLAM-7b-r']
  1|bitagent | 2024-11-17 23:26:12.529 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Generating tasks
  1|bitagent | 2024-11-17 23:28:21.793 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Generated 1000 tasks of 1000 total
  1|bitagent | 2024-11-17 23:28:21.793 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Running tasks for model Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:28:21.939 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Starting server for model Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:28:21.941 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Started server for model Salesforce/xLAM-7b-r, waiting for it to start on port 8028 (could take several minutes)
  1|bitagent | 2024-11-17 23:29:25.469 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Server for model Salesforce/xLAM-7b-r started
  1|bitagent | 2024-11-17 23:29:25.470 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Getting LLM responses for model Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:38:54.257 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Got 1000 LLM responses for model: Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:38:54.258 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Terminating server for model: Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:38:54.965 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Terminated server for model: Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:38:55.030 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Processing rewards for model: Salesforce/xLAM-7b-r, for miners: [160]
  1|bitagent | 2024-11-17 23:38:58.530 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE Scattered rewards: [np.float64(0.16442660138718893)]
  1|bitagent | 2024-11-17 23:38:58.531 |      DEBUG       | bittensor:loggingmachine.py:437 | Updated moving avg OFFLINE scores for Competition 1-1: [-0.5       -0.5       -0.5       -0.5       -0.5       -0.5
  1|bitagent | 2024-11-17 23:38:58.533 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Deleting model from HF cache: Salesforce/xLAM-7b-r
  1|bitagent | 2024-11-17 23:39:01.198 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Model 'Salesforce/xLAM-7b-r' has been removed from the cache.
  1|bitagent | 2024-11-17 23:39:01.199 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Finished processing offline tasks
  1|bitagent | 2024-11-17 23:39:02.765 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Starting offline mode for competition 1-1
  1|bitagent | 2024-11-17 23:39:03.147 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Starting offline task
  1|bitagent | 2024-11-17 23:39:03.638 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: Miner HF model names: [None]
  1|bitagent | 2024-11-17 23:39:03.638 |      DEBUG       | bittensor:loggingmachine.py:437 | OFFLINE: No unique miner HF model names to evaluate in OFFLINE mode
  ```
- If you're seeing all of this output, your validator is working!

#### Validator Hardware Requirements

Validators have hardware requirements. A series of LLMs must be loaded via sglang for evaluation:
  - Each LLM is a miner's tool calling model fetched from Hugging Face, one at a time to be evaluated OFFLINE for FINETUNED SUBMISSION and takes up 25GB to 32GB of VRAM.
  - See [min_compute.yml](./min_compute.yml).

### Miner
If you just want to run the miner without the [script](./scripts/setup_and_run.sh) or are connecting to mainnet:
```bash
# for testing, run a local validator
python3 neurons/miner.py --netuid <local netuid> --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>
# or run your model against [BFCL's evaluations](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#running-evaluations) - 
# for mainnet
pm2 start neurons/miner.py --interpreter python3 --
    --netuid 20
    --subtensor.network <finney/local/test>
    --wallet.path <YOUR PATH: e.g., ~/.bittensor/wallets> # 8.2.0 has a bug that requires wallet path to be provided
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --miner-hf-model-name-to-submit Salesforce/xLAM-7b-r # submit your own fine tune with this param
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --log_level trace # for trace logs
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine

```
#### Miner Hardware Requirements
Miners will only need to submit a top tool calling LLM or a fine-tune of their own, needing GPU with adequqate VRAM (GBs) to build their model.
There is no longer inference on SN20, so no model needs to run 24/7.

#### Default Miner
The default miner is all you need with this modification:
1) `--miner-hf-model-name-to-submit` - set this to the HF model path and repo name from Hugging Face (HF).  \
   Example: `--miner-hf-model-name-to-submit Salesforce/xLAM-7b-r`

See [Miner Configuration Considerations](#miner-configuration-considerations) for common areas miners should look to improve.

#### Miner Emissions

Miner emissions are based soley on FINETUNED SUBMISSION evaluation:
- 100% of the miner's score is determined by the persistent abilty of their FINETUNED SUBMISSION to be evaluated by the validators on the validators' machines.
- Miners' submissions are locked in and cannot be changed after submission.  Miners may, of course, register their new model as a new miner.
- New datasets are pushed out to ensure miners are not overfitting and will be used by the validators to reevaluate the miners' models.
  
The miners must finetune an 8B model (or less) to perform well on tasks similar to [BFCL Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html). Miners must publish their model to HuggingFace and update their `--miner-hf-model-name-to-submit` parameter when starting their miner - see [Default Miner](#default-miner)

#### Miner Configuration Considerations
The default miner is all you need, just make sure you update the parameters described in [Default Miner](#default-miner).  
For your consideration:
1) Use pm2 to launch your miner for easy management and reconfiguration as needed.
2) We use [SGLang](https://sgl-project.github.io/start/install.html) to run your hugging face models, please make sure your model loads with SGLang.
3) See how [we are running sglang](https://github.com/RogueTensor/bitagent_subnet/blob/e0e09470bb10b27779c0c00f9ebaabe016505832/bitagent/validator/offline_task.py#L309) to make sure you run it the same way in your testing.
4) Don't make it obvious to other miners where your HuggingFace submission is, manage this discretely.

#### Example Task
Here's an example task you can expect your model to see in FINETUNED SUBMISSION mode:

Your submitted model will receive messages like this:
```bash
[{"content":"What is the discounted price of the jacket, given it was originally $200 and there is a 20% reduction?","role":"user"}]
```
and Tools like this:
```bash
[{"arguments":{"discount_percentage":{"required":true,"type":"number","description":"The percentage discount to be applied"},
"original_price":{"description":"The original price of the item","required":true,"type":"number"}},
"description":"Calculate the discounted price of an item based on the original price and discount percentage","name":"calculate_discount"},
{"arguments":{"pod_name":{"description":"The name of the pod to be restarted","required":true,"type":"str"}},
"description":"A function to restart a given pod, useful for deployment and testing.","name":"restart_pod"},...]
```

In response, your model should return the function call like this (with the parameter values pushed in):\
`calculate_discount(discount_percentation=..., original_price=...)`

The model is responsible for returning a function call like above with the right function name, the correct function argument names and values, being sure to set any required arguments appropriately.

#### Miner Feedback
All task evaluations are pushed up to mainnet's wandb:
- WandB Mainnet - https://wandb.ai/bitagentsn20/mainnet

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
**Q: How much GPU (VRAM) and RAM do I need to run a validator and/or miner?** \
A: Validators need a GPU and require a minimum of 48 GBs of VRAM with performant CPU.  \
Miners are left to their own setup, but should be aware that the more capable tool calling LLMs require a decent amount of VRAM to fine-tune the model to be submitted.  Reminder that miners are not required to host a model for real-time inference.

**Q: Why is my miner suddenly performing poorly?** \
A: New datasets are periodically pushed to combat overfitting to any particular dataset.  \
Miners' models are locked in upon submission, therefore miners should find and finetune on larger function calling datasets to have the best chance of surviving multiple dataset pushes.\
The most ideal dataset to target is BFCL's.

**Q: Are there any required subscriptions or paid APIs?** \
A: No - no subs, no external companies, in fact we'd prefer the community build amazing AI capabilities rather than relying on corporations.

**Q: What LLM should I use?** \
A: This is where the miner needs to experiment some and test and fine-tune different LLM models to find what accomplishes the tasks most successfully.  Have a look at models in the Salesforce xLAM family as good starting points.  By this time, we may have a top performing model released on BFCL that you can start from.

**Q: Validators are running miner-submitted HF models, will validators require `trust_remote_code`?** \
A: No, we require that no setup scripts or any code be necessary for running the models.

**Q: I started my miner and I am not receiving any tasks.** \
A: There are a few things to check:
- Is your axon port, as reported on the metagraph, correct (you can check taostats or metagraph)?
- Is your axon port open and reachable from a system in the real world (like where the validators are)?
- Do you have TRACE logging on to see the dendrite requests and DEBUG logging on to see the task results?
- Make sure there isn't a stale process that is preventing your new miner process from starting up on the intended port.

**Q: What about model copying?** \
A: https://discord.com/channels/799672011265015819/1194736998250975332/1302870011362279514

**Q: My model is not being evaluated OFFLINE for FINETUNED SUBMISSION and is receiving a score of 0.** \
A: There are a few things to check:
- Is your model licensed under the apache-2.0 license?
- Is your model size less than 10B parameters? We are looking for 8B params or less models.
- Is your model name properly set in Hugging Face?

**Q: I'm getting a wallet path error, like: `KeyFileError: Keyfile at: ${HOME}/~/.bittensor/wallets/...`** \
A: There is a bug in 8.2.0 that is setting the wallet path incorrectly, so you may need to fix this by adding this parameter to your start command: \
  `--wallet.path ~/.bittensor/wallets`

**Q: I have a complicated CUDA Device setup and need to use a specific GPU device as a validator running the FINETUNED models:** \
A: We provide two parameters for this: \
  `--neuron.visible_devices`\
  `--neuron.device`\
Example usage: To use the 2nd CUDA Device, you would add these to your parameters: \
  `--neuron.visible_devices 1 --neuron.device cuda:0`

**Q: My validator is running out of GPU memory when loading OFFLINE models via sglang.** \
A: You can use this parameter: `--validator-hf-server-mem-fraction-static` to increase or decrease the amount of the GPU VRAM to use.\
It defaults to 0.85 of the VRAM.

**Q: My vTrust is low and it looks like I'm not setting OFFLINE weights.**\
A: Please test your sglang setup - check [here](#sglang-setup-for-validators).

**Q: I'm validating and seeing errors like:**
- TimeoutError
- ClientConnectorError 

A: These are responses likely during the GetHFModelName() query, they are just letting you know that the miner is not responding or connecting in time. If these are one-offs, then nothing the validator needs to do.

**Q: My validator is hanging, just printing out "Validator running ..."**\
A: There are a few things to check:
- You may not see much unless you turn on some logging, you can add this to your params to see more details:\
  `--log_level trace --logging.trace --logging.debug`
- Check your storage, make sure you didn't run out:\
  `df -h`
- If all else fails, [reach out](https://discord.com/channels/799672011265015819/1194736998250975332)


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
