import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
from typing import Union, OrderedDict
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from my_scripts.connect_mongo import get_client
from bson.objectid import ObjectId
import datetime 

import asyncio
from bson.objectid import ObjectId
from my_scripts.prepare_config import prepare_config
from my_scripts.upload_to_r2 import CloudflareR2Service
import time

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


def main():
    parser = argparse.ArgumentParser()

    taskId = os.getenv("TASK_ID")

    # asyncio.run(prepare_config(taskId))


    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    args = parser.parse_args()

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1

            taskId = os.getenv("TASK_ID")
            task = get_client()['tasks'].find_one({"_id": ObjectId(taskId)})
            model = get_client()['models'].find_one({"taskId": ObjectId(taskId)})

            # check if need to upload lora
            loraKey = f"{task['userId']}/{model['_id']}.safetensors"
            cloudflare_service = CloudflareR2Service()

            if (task['result'] is None):
                file_path = f"{os.getenv('ROOT_FOLDER')}/output/{taskId}/{taskId}.safetensors"

                if os.path.exists(file_path):
                    print("File exists.")
                else:
                    print("File does not exist.")
                    raise 

                buffer = get_file_as_buffer(file_path)


                cloudflare_service.store_file(
                    bucket_name='loras',
                    content_type='TENSOR',
                    key=loraKey,
                    buffer=buffer
                )
            for attempt in range(3):
                try:
                    # Attempt to update the task in the database
                    get_client()['tasks'].update_one(
                        {"_id": ObjectId(taskId)}, 
                        {
                            "$set": {
                                "processingStatus": 'completed',  
                                "updatedAt": datetime.datetime.now(),
                                "progress": 100,
                                "completed": True,
                                "processingCompletedAt": datetime.datetime.now(),
                                "result": {
                                    "completedIn": int((datetime.datetime.now() - task['processingStartedAt']).total_seconds() * 1000),
                                    "modelUrl": f"https://loras.cheeryclick.com/{loraKey}",
                                    "locationInfo": {
                                        "bucketName": "loras",
                                        "optimizedKeys": [],
                                        "originalKeys": [loraKey]
                                    }
                                },
                            }
                        }
                    )
                    print("Update successful.")
                    return  # Exit function if the update was successful

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < 3 - 1:
                        time.sleep(5)  # Wait before retrying
                    else:
                        print("All retries failed.")
                        raise  # Re-raise the exception after all retries have failed
                    get_client()['tasks'].update_one({"_id": ObjectId(taskId)}, {
                        "$set": {
                            "processingStatus": 'completed',  
                            "updatedAt": datetime.datetime.now(),
                            "progress": 100,
                            "completed": True,
                            "processingCompletedAt": datetime.datetime.now(),
                            "result": {
                                "completedIn": int((datetime.datetime.now() - task['processingStartedAt']).total_seconds() * 1000),
                                "modelUrl": f"https://loras.cheeryclick.com/{loraKey}",
                                "locationInfo": {
                                    "bucketName":"loras",
                                    "optimizedKeys":[],
                                    "originalKeys":[loraKey]
                                }
                            },
                        }
                    })
                for attempt in range(3):
                    try:
                        # Attempt to update the task in the database
                        get_client()['models'].update_one({"_id": ObjectId(model['_id'])}, {
                            "$set": {
                                "status": 'ready',  
                            }
                        })
                        print("Update successful.")
                        return  # Exit function if the update was successful

                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed with error: {e}")
                        if attempt < 3 - 1:
                            time.sleep(5)  # Wait before retrying
                        else:
                            print("All retries failed.")
                            raise  # Re-raise the exception after all retries have failed
                        get_client()['models'].update_one({"_id": ObjectId(model['_id'])}, {
                            "$set": {
                                "status": 'ready',  
                            }
                        })


        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                taskId = os.getenv("TASK_ID")
                task = get_client()['tasks'].find_one({"_id": ObjectId(taskId)})

                if task is None:
                    raise ValueError(f"Task with ID '{taskId}' not found.")

                get_client()['tasks'].update_one({"_id": ObjectId(taskId)}, {
                    "$set": {
                        "processingStatus": 'failed',  
                        "updatedAt": datetime.datetime.now(),  
                        "processingCompletedAt":  datetime.datetime.now(),
                        "completed": False,
                        "trainingLog": e
                    }
                })
                get_client()['models'].update_one({"taskId": ObjectId(taskId)}, {
                    "$set": {
                        "status": 'failed',  
                    }
                })             

                raise e


if __name__ == '__main__':
    main()

def get_file_as_buffer(file_path):
    with open(file_path, 'rb') as file:
        buffer = file.read()
    return buffer