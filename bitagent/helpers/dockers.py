import os
import docker
import bittensor as bt

def create_container(container_name, model_name, docker_vllm_port):
    container_to_run = "docker.io/vllm/vllm-openai:latest"

    dclient = docker.from_env()

    try:
        dclient.containers.get(container_name).remove(force=True)
    except:
        pass

    # get home directory
    home_dir = os.path.expanduser('~')

    bt.logging.debug('starting container')
    dclient.containers.run(container_to_run, 
        f"--model {model_name} --max-model-len 8912 --gpu-memory-utilization 0.9",
        name=container_name,
        device_requests=[docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])],
        detach=True,
        volumes={f'{home_dir}/.cache/huggingface': {'bind': '/root/.cache/huggingface', 'mode': 'rw'}},
        ports={'8000/tcp': docker_vllm_port})
    bt.logging.debug('started container')

    return dclient.containers.get(container_name)

def wait_for_container(openai_client, model_name):
    bt.logging.debug('waiting for container')
    while True:
        try:
            openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            break
        except Exception as e:
            #bt.logging.debug(e)
            import time
            time.sleep(1)
            pass
    bt.logging.debug('container ready')