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

import time
import uvicorn
from typing import List
from fastapi.responses import JSONResponse
from fastapi import Request, FastAPI, status, HTTPException, BackgroundTasks
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.results import Result

from bitagent.task_api.tasks import get_random_task, evaluate_task, get_organic_task
from bitagent.task_api.initiation import initiate_validator
from bitagent.protocol import QnATask

import bittensor as bt

redis = Redis(host='localhost', port=14000)
queue = Queue(connection=redis)

app = FastAPI()

# Whitelisted IPs with access to task api
# get TOP miners for organic traffic
have_data = False
while not have_data:
    try:
        subtensor = bt.subtensor(network='finney')
        subnet20 = subtensor.metagraph(netuid=20)

        vali_uids = subnet20.uids[(subnet20.S > 20000) & subnet20.validator_permit].tolist()
        vali_ips = []
        for uid in vali_uids:
            axon_ip = subnet20.addresses[uid].split("/")[-1].split(":")[0]
            if axon_ip != "0.0.0.0":
                vali_ips.append(axon_ip)
        WHITELISTED_IPS = [*vali_ips]
        bt.logging.debug("Access allowed by these IPs: ", WHITELISTED_IPS)

        # top miner uids for organic traffic
        miner_uids = subnet20.uids[(subnet20.I > 0.004) & ~subnet20.validator_permit].tolist()
        bt.logging.debug('got miner uids: ', miner_uids)

        have_data = True
    except Exception as e:
        bt.logging.error("Exception is raised when looking for top miners: ", e)

# make sure the requests are from the WHITELISTED_IPS from above
@app.middleware('http')
async def validate_ip(request: Request, call_next):
    # Get client IP
    ip = str(request.client.host)
    
    # Check if IP is allowed
    if ip not in WHITELISTED_IPS:
        data = {
            'message': f'IP {ip} is not allowed to access this resource.'
        }
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=data)

    # Proceed if IP is allowed
    return await call_next(request)

# don't need the full Validator class, but do need a couple pieces
class ValidatorStub:
    def __init__(self):
        self.device = "cuda:0"
        # use time for comparable results across validators
        def random_seed():
            return int(str(int(time.time()))[:-2])
        self.random_seed = random_seed

validator = ValidatorStub()
initiate_validator(validator)

# TODO better tasks handling
# remove old tasks
# use a queue and only keep last several
# worried this could fill up memory
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
        synapse = QnATask(prompt=response["prompt"], urls=response["urls"], datas=response["datas"], response=response["response"])

        result = evaluate_task(validator, task, synapse, response)
        return {"result": result}
    except Exception as e:
        bt.logging.error(f"Error with a task id during evaluation: {e}")
        return {"result": "Could not find task"}

# used to get an organic task if available or a generated task if no organic task
@app.post("/task_api/get_new_task")
async def get_new_task(request: Request):
    request_data = await request.json()

    # handle optional task selection via API call
    task_id_to_get = None
    sub_task_id_to_get = None
    if "task_id" in request_data.keys(): # task id (optional)
        task_id_to_get = int(request_data["task_id"])
    if "sub_task_id" in request_data.keys(): # sub level task (optional)
        sub_task_id_to_get = int(request_data["sub_task_id"])
    
    # first try to get an organic task
    task = get_organic_task()

    if not task:
        # get a generated task
        task = get_random_task(validator, task_id_to_get, sub_task_id_to_get)
        # store task via task id for evaluation - only for generated tasks
        tasks[task.task_id] = task

    return {"miner_uids": miner_uids, "task": task.toJSON()}

@app.post("/task_api/organic_response")
async def organic_response(request: Request):
    request_data = await request.json()

    try:
        bt.logging.debug(f"Received organic response: {request_data}")
        job = Job.fetch(request_data['task_id'], connection=redis)
        response = request_data["response"]
        result = {'datas': response["citations"], 'response': response["response"]}
        job.set_status('finished')
        Result.create(job, Result.Type(1), 200000, result)
        return True
    except Exception as e:
        bt.logging.error(f"Error with a task id when fetching organic task: {e}")
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
    uvicorn.run("task_api:app", host="127.0.0.1", port=8200, reload=False)
