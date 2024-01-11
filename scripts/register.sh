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
#   ./scripts/register.sh --coldkey roguetensor.main.cold --password "<password with special chars escaped>" --hotkeys roguetensor.main.hot roguetensor.main.hot2
# NOTE if you pass in the password here, you'll want to put a space in front of the command (like above) so that it won't save to history

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
        expect -c "
            spawn btcli subnet register --wallet.name $coldkey --wallet.hotkey $hotkey --subtensor.network finney --netuid $netuid
            expect -re \"want to continue?\" {send \"y\r\";}
            expect -re \"password to unlock key:\" {send \"$password\r\";}
            expect -re \"register on subnet:$netuid\" {send \"y\r\"; interact} 
        "
    done
done
