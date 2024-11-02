<div align="center">

# **BitAgent Subnet (#20) on Bittensor** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.com/channels/799672011265015819/1175085112703078400)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Transforming Your World Through Natural Language <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---
- [Introduction](#introduction)
- [Get Running](#get-running)
  - [BitAgent](#bitagent)
  - [Validator](#validator)
    - [Hardware Requirements](#hardware-requirements)
  - [Miner](#miner)
    - [Default Miner](#default-miner)
    - [Miner Considerations](#miner-considerations)
    - [Miner Feedback](#miner-feedback)
  - [Advanced](#advanced)
    - [Custom Miner](#custom-miner)
- [FAQ](#faq)
- [License](#license)

## Introduction

**Quick Pitch**: BitAgent revolutionizes how you manage tasks and workflows across platforms, merging the capabilities of large language models (LLMs) with the convenience of your favorite apps such as web browsers, Discord, and custom integrations. BitAgent empowers users to seamlessly integrate intelligent agents, providing personalized assistance and integrated task automation.

**Key Objective** - provide intelligent agency to simplify and automate tasks in your day-to-day

**See our living document for more information** - https://docs.google.com/document/d/1QVCzDu0eMmkdglD65F_Q_UjnCJauVEr62WgG8SgACt0/edit

**GoGoAgent - Our Application** - https://gogoagent.ai

**Key Features**
- OFFLINE evaluation of tool calling language model fine tunes
- ONLINE evaluation of miners running tool calling language models allowing applications to scale on top of SN20
- Working our way up the [Berkeley Fucntion Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard) (BFCL)
- No API / subscription requirements
- Run light models (8B parameter) for huge impact
- Miner's receive [transparent feedback](#miner-feedback)
- And a BONUS for getting this far - are you tired of waiting for registration slots?  Check out [register.sh](./scripts/register.sh)

---

## Get Running

- BitAgent is a competitive subnet, meaning miners succeed and fail based on how well they perform on tasks.
- **Make sure to test your miner on Testnet 76 before ever considering registering for Subnet 20.**
- Newly registered miners will start at the median score per validator and go up or down depending on their performance.
- Before getting too far, please make sure you've looked over the [Bittensor documentation](https://docs.bittensor.com/) for your needs.
- The min compute requirements are [noted below for Validators](#hardware-requirements).
- See [FAQ](#faq) for a few more details related to computing requirements for validators and miners.
- The minimum requirements for a miner are determined by the resources needed to run a competitive and performant tool calling LLM.

### BitAgent
This repository requires python3.10 or higher. 
To install and get running, simply clone this repository and install the requirements.
```bash
git clone https://github.com/RogueTensor/bitagent_subnet
cd bitagent_subnet
python -m pip install -e .
```

Then make sure to register your intended wallet (coldkey, hotkey) to Subnet 20:
```bash
btcli subnet register --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network finney --netuid 20
```

### Validator

Install [PM2](https://pm2.io/docs/runtime/guide/installation/) and the [`jq` package](https://jqlang.github.io/jq/) on your system.\
   **On Linux**:
   ```bash
   sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
   ``` 
   **On Mac OS**
   ```bash
   brew update && brew install jq && brew install npm && sudo npm install pm2 -g && pm2 update
   ```

If you want to run the validator without the [script](./scripts/setup_and_run.sh) or are connecting to mainnet:
```bash
# for testing
python3 neurons/validator.py --netuid 76 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>

# for mainnet
pm2 start neurons/validator.py --interpreter python3 -- --netuid 20 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --axon.port <PORT>

# for mainnet with AUTO UPDATES (recommended)
pm2 start run.sh --name bitagent_validators_autoupdate -- --wallet.name <your-wallet-name> --wallet.hotkey <your-wallet-hot-key> --netuid 20
```

Validators must spin-up their own LLM (specifically mistral 7B).
Note: Previously we ran the LLM's inside the validator code with the transformer package, however we pivoted away from that due to the inefficiency of running the model using vanilla transformers. Hosting the models using llama.cpp, oobabooga, vllm, TGI, are much better options as they provide additional functionality.  

To run with vLLM you can do the following:

`sudo docker run -d -p 8000:8000  --gpus all --ipc host --name mistral-instruct docker.io/vllm/vllm-openai:latest --model thesven/Mistral-7B-Instruct-v0.3-GPTQ --max-model-len 8912 --quantization gptq --dtype half`

This will run the LLM on port 8000. To change the port, change the host port for this parameter up above `-p <host port>:<container port>`.

#### Hardware Requirements

Validaotrs have hardware requirements. Two LLMS are needed to be run simultaneously:
  - 1st LLM `thesven/Mistral-7B-Instruct-v0.3-GPTQ` can run off of 10GB to 20GB of VRAM - this model is used to alter tasks before going out to miners.
  - 2nd LLM is each miner's tool calling model fetched from Hugging Face, 1 at a time to be evaluated OFFLINE and takes up 20GB to 30GB of VRAM.

Miners will need to run a top tool calling LLM or a fine-tune of their own, needing a GPU with 20GB to 30GB of VRAM. 

### Miner
If you just want to run the miner without the [script](./scripts/setup_and_run.sh) or are connecting to mainnet:
```bash
# for testing (use testnet 76)
python3 neurons/miner.py --netuid 76 --subtensor.network test --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY>
# for mainnet
pm2 start neurons/miner.py --interpreter python3 -- --netuid 20 --subtensor.network <LOCAL/FINNEY/TEST> --wallet.name <COLDKEY> --wallet.hotkey <HOTKEY> --axon.port <PORT>
```

#### Default Miner
The default miner is all you need with these modifications:
1) `--miner-hf-model-name-to-submit` - set this to the HF model path and repo name from Hugging Face (HF).  Example: `--miner-hf-model-name-to-submit Salesforce/xLAM-7b-r`
2) `--hf-model-name-to-run` - this is the model the miner is running to respond to queries to the miner. Example: `--hf-model-name-to-run Salesforce/xLAM-7b-r`
3) `--openai-api-base` - this sets the VLLM endpoint that's running your local model. Example: `--openai-api-base http://localhost:8000/v1`

See [Miner Considerations](#miner-considerations) for common areas miners should look to improve.

#### Miner Considerations
The default miner is all you need, just make sure you update the parameters described in [Default Miner](#default-miner).  
For your consideration:
1) Use VLLM as a fast inference runner for your tool calling LLM. See section on [VLLM setup](#vllm-setup).
2) Use pm2 to launch your miner for easy management and reconfiguration as needed.

#### Miner Feedback
As a miner, you receive tasks, you get rewarded, but on most subnets, you do not know what you're being graded on.
BitAgent (SN20) offers transparent feedback (in debug logging mode), so you know what you're up against.

Here's an example of a well performed task:
![miner feedback - good example](./docs/examples/output_to_miner.png)

Here's an example of a poorly performed task:
![miner feedback - bad example](./docs/examples/bad_output_to_miner.png)

Additionally, we send all queries and results to Wandb:
- WandB Testnet - https://wandb.ai/bitagentsn20/testnet
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
Q: How much GPU (VRAM) and RAM do I need to run a validator and/or miner?\
A: Validators need a GPU and require a minimum of 48 GBs of VRAM with performant CPU.  Miners are left to their own setup, but should be aware that the more capable tool calling LLMs require a decent amount of VRAM (common configurations: a 3090 (with 24GB VRAM) is capable enough for the smaller (~8B params) models we require).

Q: Are there any required subscriptions or paid APIs?\
A: No - no subs, no external companies, in fact we'd rather the community build amazing AI capabilities than relying on corporations.

Q: What LLM should I use?\
A: This is where the miner needs to experiment some and test and fine-tune different LLM models to find what accomplishes the tasks most successfully.  Have a look at models in the Salesforce xLAM family as good starting points.

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
