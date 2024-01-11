#!/bin/bash
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

#example:
#   ./scripts/register.sh --coldkey <YOUR COLDKEY> --password "<password with special chars escaped>" --hotkeys <YOUR HOTKEY1> <YOUR HOTKEY2>
# NOTE if you pass in the password here, you'll want to put a space in front of the command (like above) so that it won't save to history
# RISK ^^^ look at the NOTE above

#########################
# WHAT THIS SCRIPT DOES #
# USE AT YOUR OWN RISK  #
#########################
# You need a wallet for a coldkey, that's YOUR COLDKEY
# You need some amount of Tao in that wallet, enough to register to the subnet you want to register to
# THEN
# Pass in the hotkey(s) you want to register (they don't have to exist - we'll create them for you) [See example above]
# Let the script do the rest
# -- sends "y" keys when needed to confirm registration
# -- sends your password when needed to decrypt the wallet/send tao
# -- keeps trying until you stop it (even if it's successfully registered)
# RISK: this script is unsupervised and WILL spend your Tao, please review and determine for yourself if this is a RISK you are willing to take

############ GET THE ARGS ############
programname=$0
function usage {
    echo ""
    echo "Creates wallets for the subnet (owner, validators, miners), funds them, registers them, then starts them."
    echo ""
    echo "usage: $programname"
    echo ""
    echo "  --coldkey        string  coldkey"
    echo "                           (required)"
    echo "  --hotkeys        array   list of hotkeys"
    echo "                           (required)"
    echo "  --password       string  decrypt pw"
    echo "                           (required)"
    echo "  --netuid                 the netuid to work with"
    echo "                           (default: 20)"
    echo ""
}

hotkeys=()  # Declare hotkeys as an array

while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "-h" ]]; then
        usage
        exit 0
    elif [[ $1 == "--hotkeys" ]]; then
        shift  # Shift past the '--hotkeys'
        while [[ $1 && ${1:0:2} != "--" ]]; do
            hotkeys+=("$1")  # Add to the hotkeys array
            shift  # Shift past the value
        done
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        v="${v//-/_}"  # Replace hyphens with underscores

        # Check if the next argument is a value or another option
        if [[ $2 && ${2:0:2} != "--" ]]; then
            declare "$v"="$2"
            shift  # Shift past the value
        else
            declare "$v"=0  # Set a default value (true) for flags without a specific value
        fi
    fi
    shift
done

echo $hotkeys
netuid=${netuid:-20}

############ REGISTER to the SUBNET ###################
while true
do
    for hotkey in "${hotkeys[@]}"; do
        # if hotkey does not exist, create it
        if [ ! -f ~/.bittensor/wallets/${coldkey}/hotkeys/$hotkey ]; then
            echo "#######################################################################################"
            echo "$hotkey not found! Creating it under $coldkey. Make sure to grab the mnemonic."
            echo "NOTE: mnemonic info will be logged to mnemonics.txt"
            echo "WARNING: make sure to clear out the mnemonics.txt file and don't leave it on the system"
            echo "#######################################################################################"
            btcli w new_hotkey --wallet.name $coldkey --wallet.hotkey $hotkey 2>&1 >> mnemonics.txt
        fi

        expect -c "
            spawn btcli subnet register --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network finney --netuid $netuid
            expect -re \"want to continue?\" {send \"y\r\";}
            expect -re \"password to unlock key:\" {send \"$password\r\";}
            expect -re \"register on subnet:$netuid\" {send \"y\r\"; interact} 
        "
    done
done
