import os
import yaml
import argparse
from types import SimpleNamespace

"""
Usage:
    python launcher.py --config_file configs/XXX.yaml
"""

# vllm server cmd template
server_cmd = """
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_SAMPLER=1
vllm serve {HF_MODEL_FOLDER} \\
    --port {PORT} \\
    --dtype bfloat16 \\
    --disable-custom-all-reduce \\
    --tensor-parallel-size {TP} \\
    --pipeline-parallel-size {PP} \\
    --max-model-len {MAX_SEQ_LEN} \\
    --enforce-eager
"""

# evaluator cmd template
# run multiple evaluators in parallel
evaluator_cmd = """
echo "Sleeping for {SLEEP_SECS} seconds"
sleep {SLEEP_SECS}
echo "Starting evaluator"
export PYTHONPATH={ROOT_FOLDER}/.:${{PYTHONPATH}}
client_pids=()
cd {ROOT_FOLDER}/evaluators
for ((i={VOTER_ID_START}; i<{VOTER_ID_END}; i++)); do
    python evaluator.py \\
        --config_file {CONFIG_FILE} \\
        --voter_id ${{i}} \\
    &
    client_pids+=($!)
done
for pid in "${{client_pids[@]}}"; do
    wait $pid
done
"""

# aggregator cmd template
aggregator_cmd = """
export PYTHONPATH={ROOT_FOLDER}/.:${{PYTHONPATH}}
cd {ROOT_FOLDER}/evaluators \\
&& python aggregator.py \\
    --config_file {CONFIG_FILE}
"""

# srun cmd template
srun_cmd = """
srun \\
  --output={LOG_FILE} \\
  --error={LOG_FILE} \\
  --container-image={CONTAINER_PATH} \\
  --container-mounts={MOUNTS} \\
  --ntasks-per-node=1 \\
  --nodes={NODES} \\
  --overlap \\
  --overcommit \\
  bash -c '{CMD}'
"""

def run_local(config):
    """run locally: run .sh in ./scripts/config_version/"""
    # format server cmd
    server_file = os.path.join(config.scripts_folder, "run_local_server.sh")   
    server_log = os.path.join(config.logs_folder, "server.log")
    server_local_cmd = server_cmd.format(
        HF_MODEL_FOLDER=config.model.model_name_or_path,
        PORT=config.model.api_port,
        TP=config.model.tp,
        PP=config.model.pp,
        MAX_SEQ_LEN=config.model.max_seq_len,
    )
    # create and execute the server script
    with open(server_file, "w") as f:
        f.write("#! /bin/bash\n")
        f.write(server_local_cmd)
    os.chmod(server_file, 0o755)  # executable script
    os.system(f"bash {server_file} > {server_log} 2>&1 &")  # run server in background
    print("Running server: ", server_file)

    # format evaluator and aggregator cmds
    client_file = os.path.join(config.scripts_folder, "run_local_client.sh")
    client_log = os.path.join(config.logs_folder, "client.log")
    evaluator_local_cmd = evaluator_cmd.format(
        SLEEP_SECS=config.eval.client_sleep,
        ROOT_FOLDER=config.eval.root_folder,
        CONFIG_FILE=config.config_file,
        VOTER_ID_START=0,
        VOTER_ID_END=config.eval.num_voters_per_server,
    )
    aggregator_local_cmd = aggregator_cmd.format(
        CONFIG_FILE=config.config_file,
        ROOT_FOLDER=config.eval.root_folder,
    )
    # create and execute the client script
    # 1. run voters in parallel
    # 2. wait for all voters to finish
    # 3. run aggregator    
    with open(client_file, "w") as f:
        f.write("#! /bin/bash\n")
        f.write(evaluator_local_cmd)
        f.write(aggregator_local_cmd)   
    os.chmod(client_file, 0o755)  # executable script
    os.system(f"bash {client_file} > {client_log} 2>&1 &")  # run client in background
    print("Running client: ", client_file)

def run_slurm(config):
    """run on slurm: run .sh in ./scripts/config_version/"""
    
    # launch multiple server-evaluator pairs as separate SLURM jobs
    for i in range(config.eval.slurm_jobs):
        # setup server job
        slurm_eval_file = os.path.join(config.scripts_folder, f"run_slurm_job_eval_{i}.sh")
        server_log = os.path.join(config.logs_folder, f"{i}_server.log")
        server_srun_cmd = srun_cmd.format(
            LOG_FILE=server_log,
            CONTAINER_PATH=config.eval.container_path,
            MOUNTS=",".join(config.eval.slurm_mounts),
            NODES=config.eval.slurm_nodes,
            CMD=server_cmd.format(
                HF_MODEL_FOLDER=config.model.model_name_or_path,
                PORT=config.model.api_port,
                TP=config.model.tp,
                PP=config.model.pp,
                MAX_SEQ_LEN=config.model.max_seq_len,
            ),
        )
        
        # setup evaluator job
        evaluator_log = os.path.join(config.logs_folder, f"{i}_evaluator.log")
        evaluator_srun_cmd = srun_cmd.format(
            LOG_FILE=evaluator_log,
            CONTAINER_PATH=config.eval.container_path,
            MOUNTS=",".join(config.eval.slurm_mounts),
            NODES=1,
            CMD=evaluator_cmd.format(
                SLEEP_SECS=config.eval.client_sleep,
                ROOT_FOLDER=config.eval.root_folder,
                CONFIG_FILE=config.config_file,
                VOTER_ID_START=i * config.eval.num_voters_per_server,
                VOTER_ID_END=(i + 1) * config.eval.num_voters_per_server,
            ),
        )

        # create a script that runs both server and evaluator
        with open(slurm_eval_file, "w") as f:
            f.write("#! /bin/bash\n")
            f.write("set -x\n")  
            f.write(server_srun_cmd.strip() + " &\n")       # run server in background
            f.write(evaluator_srun_cmd.strip() + " &\n")    # run evaluators and wait
            f.write("wait $!\n")                            
            f.write("set +x\n") 

        os.chmod(slurm_eval_file, 0o755)
        
        # submit job
        os.system(f"""
        sbatch \
        --output={os.path.join(config.logs_folder, "slurm-%j.out")} \
        --error={os.path.join(config.logs_folder, "slurm-%j.out")} \
        --account={config.eval.slurm_account} \
        --partition={config.eval.slurm_partition} \
        --time={config.eval.slurm_timeout} \
        --gres=gpu:8 \
        --nodes={config.eval.slurm_nodes} \
        --job-name=L0-Bench-{config.config_version} \
        --comment=metrics \
        {slurm_eval_file}
        """)
        print("Running slurm job: ", slurm_eval_file)

    # setup and launch the aggregator job
    slurm_agg_file = os.path.join(config.scripts_folder, f"run_slurm_job_aggregate.sh")     
    aggregator_log = os.path.join(config.logs_folder, f"aggregator.log")
    aggregator_srun_cmd = srun_cmd.format(
        LOG_FILE=aggregator_log,
        CONTAINER_PATH=config.eval.container_path,
        MOUNTS=",".join(config.eval.slurm_mounts),
        NODES=1,
        CMD=aggregator_cmd.format(
            CONFIG_FILE=config.config_file,
            ROOT_FOLDER=config.eval.root_folder,
        ),
    )
    
    # create the aggregator script
    with open(slurm_agg_file, "w") as f:
        f.write("#! /bin/bash\n")
        f.write("set -x\n")
        f.write(aggregator_srun_cmd)
        f.write("set +x\n")
    
    os.chmod(slurm_agg_file, 0o755)
    
    # submit job
    os.system(f"""
    sbatch \
    --output={os.path.join(config.logs_folder, "slurm-%j.out")} \
    --error={os.path.join(config.logs_folder, "slurm-%j.out")} \
    --account={config.eval.slurm_account} \
    --partition={config.eval.slurm_partition} \
    --time={config.eval.slurm_timeout} \
    --gres=gpu:1 \
    --nodes=1 \
    --job-name=L0-Bench-{config.config_version} \
    --dependency=singleton \
    --comment=metrics \
    {slurm_agg_file}
    """)
    print("Running slurm job: ", slurm_agg_file)

def load_yaml_config(config_data):
    def create_namespace(data):
        if isinstance(data, dict):
            namespace = SimpleNamespace()
            for key, value in data.items():
                setattr(namespace, key, create_namespace(value))
            return namespace
        else:
            return data
    return create_namespace(config_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    # load and process the config file
    with open(args.config_file, "r") as f:
        config = load_yaml_config(yaml.safe_load(f))
        config_version = os.path.splitext(os.path.basename(args.config_file))[0]
        config.config_file = os.path.abspath(args.config_file)
        config.config_version = config_version
    
    # create necessary dirs for scripts and logs
    scripts_folder = os.path.join(config.eval.result_folder, config.config_version, "scripts")
    logs_folder = os.path.join(config.eval.result_folder, config.config_version, "logs")
    os.makedirs(scripts_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    config.scripts_folder = scripts_folder
    config.logs_folder = logs_folder

    if config.eval.cluster == "local":
        run_local(config)
    elif config.eval.cluster == "slurm":
        run_slurm(config)
    else:
        raise ValueError(f"Invalid cluster setup: {config.eval.cluster}")
