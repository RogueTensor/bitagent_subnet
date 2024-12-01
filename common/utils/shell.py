import subprocess
import shlex

def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and stream the output to the caller in real-time.

    Args:
        command: Shell command as a string (can include \\ line continuations)
    Returns:
        subprocess.Popen: The process handle for further interaction.
    """
    # Replace \ newline with space and split using shlex
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = shlex.split(command)  # Handles quoted strings correctly

    try:
        # Run the process
        process = subprocess.Popen(
            parts, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Stream the output in real-time
        def stream_output(stream, stream_name):
            for line in iter(stream.readline, ''):
                print(f"{stream_name}: {line.strip()}")
            stream.close()

        # Stream both stdout and stderr
        from threading import Thread
        Thread(target=stream_output, args=(process.stdout, "STDOUT")).start()
        Thread(target=stream_output, args=(process.stderr, "STDERR")).start()

        return process
    except Exception as e:
        print(f"Error executing command: {command}. Exception: {e}")
        raise