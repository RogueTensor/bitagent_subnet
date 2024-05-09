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
import json
from cachetools.func import ttl_cache
import time
from contextlib import asynccontextmanager

# import uvicorn
# from uvicorn.config import LOGGING_CONFIG
from fastapi.responses import JSONResponse, ORJSONResponse

from fastapi import Request, FastAPI, HTTPException, BackgroundTasks
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.results import Result
import random
from bitagent.task_api.tasks import get_random_task, evaluate_task, get_organic_task
from bitagent.task_api.tasks.task import Task
from bitagent.task_api.initiation import initiate_validator
from bitagent.protocol import QnATask
import logging
import torch
from starlette.concurrency import run_in_threadpool
import bittensor as bt
from starlette.middleware.base import BaseHTTPMiddleware
import traceback
logging.basicConfig(level=logging.DEBUG)




    

subnet20 = None
organic_miner_uids = None
redis = None
queue = None
subtensor = None
validator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global subnet20, organic_miner_uids, redis, queue, subtensor, validator
    # Setup basic logging configuration
    redis = Redis(host="localhost", port=14000, decode_responses=True)
    queue = Queue(connection=redis)


    have_data = False
    subtensor = bt.subtensor(network='finney')
    subnet20 = subtensor.metagraph(netuid=20)
    while not have_data:
        try:

            # get TOP miners for organic traffic
            vali_uids = subnet20.uids[
                (subnet20.S > 20000) & subnet20.validator_permit
            ].tolist()
            vali_ips = []
            for uid in vali_uids:
                axon_ip = subnet20.addresses[uid].split("/")[-1].split(":")[0]
                if axon_ip != "0.0.0.0":
                    vali_ips.append(axon_ip)

            # top miner uids for organic traffic
            organic_miner_uids = subnet20.uids[
                (subnet20.I > 0.004) & ~subnet20.validator_permit
            ].tolist()
            logging.info(f"got miner uids: {organic_miner_uids}")

            have_data = True
        except Exception as e:
            logging.error(f"Exception is raised when looking for top miners:  {e}")


    # don't need the full Validator class, but do need a couple pieces
    class ValidatorStub:
        def __init__(self):
            self.device = "cuda:0"

            # use time for comparable results across validators
            def random_seed():
                None

            self.random_seed = random_seed


    validator = ValidatorStub()
    initiate_validator(validator)
    # Required to load api correctly
    validator.sentence_transformer.encode("loading")
    yield
    

@ttl_cache(maxsize=100, ttl=60 * 60)
# Whitelisted IPs with access to task api
def get_whitelisted_ips():
    """
    Returns the current whitelist. This function will be cached, and the cache
    will be automatically refreshed after the TTL expires (1hr).
    """

    subnet20.sync(subtensor=subtensor)
    vali_uids = subnet20.uids[(subnet20.S > 20000) & subnet20.validator_permit].tolist()
    vali_ips = []
    for uid in vali_uids:
        axon_ip = subnet20.addresses[uid].split("/")[-1].split(":")[0]
        if axon_ip != "0.0.0.0":
            vali_ips.append(axon_ip)
    whitelisted_ips = [
        "127.0.0.1",
        "192.168.69.194",
        "192.150.253.122",
        "213.173.108.99",
        "34.90.35.204",
        "137.184.2.196", # added "137.184.2.196" tao validatonr
        "45.76.13.68", # keith
        *vali_ips,
    ]
    # logging.info(f"Access allowed by these IPs:  {whitelisted_ips}")
    print(f"date:hour:minute: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)
    print(f"Access allowed by these IPs:  {whitelisted_ips}", flush=True)

    return whitelisted_ips


class IPFilterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        if client_ip not in get_whitelisted_ips():
            return JSONResponse(content={"detail": "Access denied"}, status_code=403)
        response = await call_next(request)
        return response


class ProcessTimeHeader(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        print(f"{request.url} request took {time.time() - start_time} sec")
        return response
app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)

app.add_middleware(ProcessTimeHeader)
app.add_middleware(IPFilterMiddleware)



@app.post("/task_api/evaluate_task_response")
async def evaluate_task_response(request: Request):
    request_data = await request.json()
    # logging.error(f"Received task response: {request_data}")
    try:
        task_id = request_data["task_id"]  # task id for lookup
        # task = tasks[task_id]
        logging.info(f"Task id for evaluation: {task_id}")
        task = json.loads(redis.get(task_id))
        task = Task.fromSerialized(task)

        response = request_data["response"]
        synapse = QnATask(
            prompt=response["prompt"],
            urls=response["urls"],
            datas=response["datas"],
            response=response["response"],
        )

        result = await run_in_threadpool(
            evaluate_task, validator, task, synapse, response
        )
        return {"result": result}
    except Exception as e:
        logging.error(f"ERROR: {e}")
        logging.error(traceback.format_exc())
        logging.error(f"Error with a task id during evaluation: {e}")
        return {"result": "Could not find task"}


# used to get an organic task if available or a generated task if no organic task
@app.post("/task_api/get_new_task")
async def get_new_task(request: Request):
    request_data = await request.json()
    # logging.error(f"Received task response: {request_data}")

    # handle optional task selection via API call
    task_id_to_get = request_data.get("task_id", None)
    sub_task_id_to_get = request_data.get("sub_task_id", None)

    task = get_organic_task()

    miner_uids = []

    if not task_id_to_get and not sub_task_id_to_get and not task:
        try:
            tasks = ["pet","qna","summary", "ansible"]
            random_task = random.choice(tasks)
            key = str(redis.get("latest-key"))
            print(f'The latest key is {key}',flush=True)
            serialized_task = redis.get(key + random_task)
            if not serialized_task:
                serialized_task = redis.get(key)
            task = json.loads(serialized_task)
            task = Task.fromSerialized(task)
            task.task_id = key + random_task
        except:
            pass

        miner_uids = []
    
    if task_id_to_get or sub_task_id_to_get or not task:
        # get a generated task
        task = await run_in_threadpool(
            get_random_task, validator, task_id_to_get, sub_task_id_to_get
        )
        # store task via task id for evaluation - only for generated tasks
        # tasks[task.task_id] = task
        #redis.set(f'task at time{time.time()}', json.dumps(task.serialize()))
        redis.set(f'{task.task_id}', json.dumps(task.serialize()))
        # no miner uids
        miner_uids = []
    else:
        miner_uids = torch.tensor(organic_miner_uids)

    return {"miner_uids": miner_uids, "task": task.toJSON()}


@app.post("/task_api/organic_response")
async def organic_response(request: Request):
    request_data = await request.json()

    try:
        logging.info(f"Received organic response: {request_data}")
        job = Job.fetch(request_data["task_id"], connection=redis)
        response = request_data["response"]
        result = {"datas": response["citations"], "response": response["response"]}
        job.set_status("finished")
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


@app.post("/process")
async def process_request(request: Request, background_tasks: BackgroundTasks):
    request_data = await request.json()
    prompt = request_data["prompt"]
    datas = request_data["datas"] if "datas" in request_data else []
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    job = queue.enqueue(process_prompt, prompt, datas)
    return {"task_id": job.id}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    job = Job.fetch(task_id, connection=redis)
    if job.is_finished:
        return {
            "last_result": job.result,
            "results": [result.return_value for result in job.results()],
        }
    else:
        return {"status": "processing"}


@app.get("/tasks")
async def get_tasks():
    jobs = queue.jobs
    return {"tasks": [job.id for job in jobs]}