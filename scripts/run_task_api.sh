docker run -p 14000:6379 -td redis
docker run -d -p 14025:8000  --gpus device=0 --ipc host --name modelname docker.io/vllm/vllm-openai:latest --model models/llm --max-model-len 8912 --quantization gptq --dtype half --gpu-memory-utilization 0.5

source env/bin/activate
pip3 install -r requirements.txt
pip3 uninstall uvloop

cd bitagent/task_api/
pm2 start task_generator.py --name task_gen --interpreter python3
pm2 start --name TaskAPI.8200 "gunicorn task_api:app --workers 3 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8200 --timeout 600 --access-logfile -"
