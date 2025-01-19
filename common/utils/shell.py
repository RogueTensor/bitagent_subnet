import shlex
import subprocess
import bittensor as bt
from threading import Thread

def execute_shell_command(command: str, model_name: str) -> subprocess.Popen:
    """
    Execute a shell command and stream the output to the caller in real-time.

    Args:
        command: Shell command as a string (can include \\ line continuations)
    Returns:
        subprocess.Popen: The process handle for further interaction.
    """
    # Replace \ newline with space and split using shlex
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = shlex.split(command)  # Handles quoted strings correct

    try:
        # Run the process
        process = subprocess.Popen(
            parts, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        def stream_output(stream, stream_name):
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n')
                if stream_name == "STDERR":
                    # log everything except for token generation metrics
                    if "#new-token" not in line and "Decode batch." not in line:
                        redacted_line = line.replace(model_name, "[REDACTED]")
                        bt.logging.debug(f"{stream_name}: {redacted_line}")

                # Uncomment this if you want STDOUT logging as well:
                # else:
                #     print(f"{stream_name}: {line}")

            stream.close()

        # Stream both stdout and stderr
        Thread(target=stream_output, args=(process.stdout, "STDOUT")).start()
        Thread(target=stream_output, args=(process.stderr, "STDERR")).start()

        return process
    except Exception as e:
        print(f"Error executing command: {command}. Exception: {e}")
        raise
