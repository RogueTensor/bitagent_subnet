# The MIT License (MIT)
# Copyright © 2024 RogueTensor

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
from cachetools.func import ttl_cache
import time
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi import Request, FastAPI, HTTPException, BackgroundTasks
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.results import Result
from starlette.middleware.base import BaseHTTPMiddleware

from bitagent.task_api.tasks import get_random_task, evaluate_task
from bitagent.task_api.initiation import initiate_validator
from bitagent.task_api.tasks.task import Task
from bitagent.protocol import QnATask
import logging
import torch

import bittensor as bt
# Setup basic logging configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
redis = Redis(host='localhost', port=14000)
queue = Queue('tasks', connection=redis)

#app = FastAPI()

have_data = False
while not have_data:
    try:
        subtensor = bt.subtensor(network='finney')
        subnet20 = subtensor.metagraph(netuid=20)
        
        # get valis for organic traffic
        vali_uids = []
        vali_ips = []
        for uid in vali_uids:
            axon_ip = subnet20.addresses[uid].split("/")[-1].split(":")[0]
            if axon_ip != "0.0.0.0":
                vali_ips.append(axon_ip)

        # top miner uids for organic traffic
        organic_miner_uids = subnet20.uids[(subnet20.I > 0.004) & ~subnet20.validator_permit].tolist()
        logging.debug(f'got miner uids: {organic_miner_uids}')

        have_data = True
    except Exception as e:
        logging.error(f"Exception is raised when looking for top miners: {e}")



@ttl_cache(maxsize=100, ttl=60*60)
# Whitelisted IPs with access to task api
def get_whitelisted_ips():
    """
    Returns the current whitelist. This function will be cached, and the cache
    will be automatically refreshed after the TTL expires (1hr).
    """
    subtensor = bt.subtensor(network="finney")
    subnet20 = subtensor.metagraph(netuid=20)

    # get valis for organic traffic
    vali_uids = []
    vali_ips = []
    for uid in vali_uids:
        axon_ip = subnet20.addresses[uid].split("/")[-1].split(":")[0]
        if axon_ip != "0.0.0.0":
            vali_ips.append(axon_ip)
    whitelisted_ips = vali_ips
    logging.debug(f"Access allowed by these IPs: {whitelisted_ips}")

    return whitelisted_ips

# Middleware for IP validation

class IPFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        if client_ip not in get_whitelisted_ips():
            return JSONResponse(content={"detail": "Access denied"}, status_code=403)
        response = await call_next(request)
        return response
app = FastAPI(default_response_class=ORJSONResponse)
app.add_middleware(IPFilterMiddleware)

class ValidatorStub:
    def __init__(self):
        self.device = "cuda:0"
        # use time for comparable results across validators
        def random_seed():
            return int(str(int(time.time()))[:-2])
        self.random_seed = random_seed
        self.subtensor = bt.subtensor(network="finney")
        self.metagraph = self.subtensor.metagraph(netuid=20)

def get_organic_task():
        jobs = [job for job in queue.jobs if job.is_queued]
        if not jobs:
            return None

        job = jobs[0]
        job.set_status('started')
        try:
            job_datas = job.args[1]
        except Exception as e:
            job_datas = []

        return Task(name="Organic Task", task_type="organic", prompt=job.args[0], datas=job_datas, task_id=job.id)

validator = ValidatorStub()
initiate_validator(validator)

tasks = {}

# used to evaluate generated tasks
# not used for organic traffic
@app.post("/task_api/evaluate_task_response")
async def evaluate_task_response(request: Request):
    request_data = await request.json()

    try:
        task_id = request_data["task_id"] # task id for lookup
        task = tasks[task_id]

        response = request_data["response"]
        task.synapse.response = response['response']
        result = evaluate_task(validator, task, task.synapse, response)
        return {"result": result}
    except Exception as e:
        logging.error(f"Error with a task id during evaluation: {e}")
        return {"result": "Could not find task"}

# used to get an organic task if available or a generated task if no organic task
@app.post("/task_api/get_new_task")
async def get_new_task(request: Request):
    request_data = await request.json()

    # handle optional task selection via API call
    task_name_to_get = None
    sub_task_id_to_get = None
    if "task_name" in request_data.keys(): # task id (optional)
        task_name_to_get = request_data["task_name"]
    if "sub_task_id" in request_data.keys(): # sub level task (optional)
        sub_task_id_to_get = int(request_data["sub_task_id"])
    
    # first try to get an organic task
    task = get_organic_task()

    miner_uids = []
    if not task:
        # get a generated task
        task = get_random_task(validator, task_name_to_get, sub_task_id_to_get)
        # store task via task id for evaluation - only for generated tasks
        tasks[task.task_id] = task
        # no miner uids
        miner_uids = []
    else:
        # is organic task, no need to store task id
        # set the miner uids for the organic task
        miner_uids = torch.tensor(organic_miner_uids)
    print(f"New task gotten: {task.toJSON()}", flush=True)
    return {"miner_uids": miner_uids, "task": task.toJSON()}

@app.post("/task_api/organic_response")
async def organic_response(request: Request):
    request_data = await request.json()

    try:
        logging.debug(f"Received organic response: {request_data}")
        job = Job.fetch(request_data['task_id'], connection=redis)
        response = request_data["response"]
        result = {'datas': response["citations"], 'response': response["response"]}
        job.set_status('finished')
        Result.create(job, Result.Type(1), 200000, result)
        return True
    except Exception as e:
        logging.error(f"Error with a task id when fetching organic task: {e}")
        return True

# Placeholder for Redis enqueue to work
def process_prompt(prompt, datas):
    # Simulating some processing
    response = f"Received prompt: {prompt}, Datas: {datas}. Processed."
    return response

@app.post('/process')
async def process_request(request: Request, background_tasks: BackgroundTasks):
    request_data = await request.json()
    prompt = request_data['prompt']
    datas = request_data['datas'] if "datas" in request_data else []
    if not prompt:
        raise HTTPException(status_code=400, detail='Prompt is required.')

    job = queue.enqueue(process_prompt, prompt, datas)
    return {'task_id': job.id}

@app.get('/result/{task_id}')
async def get_result(task_id: str):
    job = Job.fetch(task_id, connection=redis)
    if job.is_finished:
        return {'last_result': job.result, 'results': [result.return_value for result in job.results()]}
    else:
        return {'status': 'processing'}

@app.get('/tasks')
async def get_tasks():
    jobs = queue.jobs
    return {'tasks': [job.id for job in jobs]}

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = (
        "%(asctime)s " + LOGGING_CONFIG["formatters"]["access"]["fmt"]
    )
    uvicorn.run("task_api:app", host="127.0.0.1", port=8400, reload=False)
