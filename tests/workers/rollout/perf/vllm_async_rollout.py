# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compare vLLM AsyncLLM backend: ExternalRayDistributedExecutor(remote call) vs RayDistributedExecutor(compiled graph)

1. Prepare openai/gsm8k dataset
python3 examples/data_preprocess/gsm8k.py

2. Install latest ray==2.46.0
pip uninstall -y bytedray && pip install "ray[cgraph]"

3. Start ray
ray start --head

4. Run perf test
python3 tests/workers/rollout/perf/vllm_async_rollout.py >perf.log 2>&1

hardware: Nvidia 8*H20
packages:
- torch==2.6.0
- vllm==0.8.5

[DEBUG] backend: sync, n_gpus_per_node: 2, use_chat_scheduler: False, batch_size: 1024, step: 0, step_time: 45.74 secs
[DEBUG] backend: cgraph, n_gpus_per_node: 2, use_chat_scheduler: True, batch_size: 1024, step: 0, step time: 51.75 secs, wake_up_time: 0.00 secs, gen_time: 51.75 secs, sleep_time: 0.00 secs
[DEBUG] backend: external, n_gpus_per_node: 2, use_chat_scheduler: True, batch_size: 1024, step: 0, step time: 50.94 secs, wake_up_time: 1.69 secs, gen_time: 48.60 secs, sleep_time: 0.65 secs
[DEBUG] backend: external, n_gpus_per_node: 2, use_chat_scheduler: True, batch_size: 1024, step: 0, step time: 50.07 secs, wake_up_time: 1.23 secs, gen_time: 48.21 secs, sleep_time: 0.63 secs
[DEBUG] backend: external, n_gpus_per_node: 8, use_chat_scheduler: True, batch_size: 1024, step: 0, step time: 20.35 secs, wake_up_time: 0.91 secs, gen_time: 18.48 secs, sleep_time: 0.96 secs
"""

import asyncio
import os
import time
from typing import Tuple, Union
from uuid import uuid4

import aiohttp
import ray
from omegaconf import DictConfig, OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
from ray.util.placement_group import placement_group
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer


def init_config(n_gpus_per_node) -> DictConfig:
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.trainer.n_gpus_per_node = n_gpus_per_node
    config.data.train_batch_size = 1024
    config.data.return_raw_chat = True
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-7B-Instruct"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 1

    # test sleep/wake_up with fsdp offload
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True

    return config


def async_llm_backend_ray_compiled_graph(init_config) -> Tuple[ray.actor.ActorHandle, str]:
    """AsyncLLM backend RayDistributedExecutor with compiled graph."""
    pg = placement_group([{"CPU": 1}, {"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}])
    ray.get(pg.ready())
    worker = AsyncvLLMServer.options(
        scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(init_config, vllm_dp_size=2, vllm_dp_rank=0, wg_prefix="")
    ray.get(worker.init_engine.remote())
    server_address = ray.get(worker.get_server_address.remote())

    return worker, server_address


def async_llm_backend_external_ray_executor(init_config) -> Tuple[AsyncLLMServerManager, str]:
    """AsyncLLM backend ExternalRayDistributedExecutor with remote call."""
    async_rollout_manager = init_async_rollout_manager(init_config)
    return async_rollout_manager, async_rollout_manager.server_addresses[0]


async def chat_completions(address: str, **chat_complete_request) -> ChatCompletion:
    try:
        extra_body = chat_complete_request.pop("extra_body", {})
        chat_complete_request.update(extra_body or {})
        extra_headers = chat_complete_request.pop("extra_headers")
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


def initialize(config, backend) -> Tuple[Union[AsyncLLMServerManager, ray.actor.ActorHandle], str, StatefulDataLoader]:
    env_vars = {
        "NCCL_DEBUG": "WARN",
        "VLLM_USE_V1": "1",
        "VERL_VLLM_USE_RAY_BACKEND": "1" if backend == "external" else "0",
    }
    ray.init(runtime_env={"env_vars": env_vars})

    # STEP 1: init async llm server
    server, server_address = None, None
    if backend == "external":
        server = init_async_rollout_manager(config)
        server_address = server.server_addresses[0]
        server.sleep()
    elif backend == "cgraph":
        server, server_address = async_llm_backend_ray_compiled_graph(config)
    elif backend == "sync":
        server = init_async_rollout_manager(config)
        server_address = None
    print(f"server_address: {server_address}")

    # STEP 2: create dataloader
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/gsm8k/train.parquet"),
        tokenizer=tokenizer,
        config=config.data,
    )
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
        num_workers=config.data.get("dataloader_num_workers", 8),
        drop_last=True,
        collate_fn=default_collate_fn,
        sampler=SequentialSampler(dataset),
    )

    return server, server_address, dataloader


async def perf_without_chat_scheduler(backend, n_gpus_per_node):
    """Perf test AsyncLLM without chat_scheduler."""
    config = init_config(n_gpus_per_node)
    server, server_address, dataloader = initialize(config, backend)

    sampling_params = dict(
        model="/".join(config.actor_rollout_ref.model.path.split("/")[-2:]),
        temperature=config.actor_rollout_ref.rollout.temperature,
        top_p=config.actor_rollout_ref.rollout.top_p,
    )
    for step, batch in enumerate(dataloader):
        batch: DataProto = DataProto.from_single_dict(batch)
        t_start = time.time()
        if backend == "external":
            server.wake_up()
        t_gen_start = time.time()
        tasks = []
        for messages in batch.non_tensor_batch["raw_prompt"].repeat(config.actor_rollout_ref.rollout.n, axis=0):
            task = chat_completions(
                server_address,
                messages=messages.tolist(),
                extra_headers={"x-request-id": uuid4().hex},
                **sampling_params,
            )
            tasks.append(task)

        completions = await asyncio.gather(*tasks)
        messages = [completion.choices[0].message.content for completion in completions]
        t_gen_end = time.time()
        if backend == "external":
            server.sleep()
        t_end = time.time()

        wake_up_time = t_gen_start - t_start
        gen_time = t_gen_end - t_gen_start
        sleep_time = t_end - t_gen_end
        step_time = t_end - t_start
        print(f"[DEBUG] backend: {backend}, n_gpus_per_node: {n_gpus_per_node}, use_chat_scheduler: True, batch_size: {len(messages)}, step: {step}, step time: {step_time:.2f} secs, wake_up_time: {wake_up_time:.2f} secs, gen_time: {gen_time:.2f} secs, sleep_time: {sleep_time:.2f} secs")
        break

    ray.shutdown()


def perf_with_chat_scheduler(backend, n_gpus_per_node):
    """Perf test AsyncLLM with chat_scheduler."""
    assert backend == "external"

    config = init_config(n_gpus_per_node)
    server, server_address, dataloader = initialize(config, backend)
    for step, batch in enumerate(dataloader):
        batch: DataProto = DataProto.from_single_dict(batch)
        t_start = time.time()
        server.wake_up()
        t_gen_start = time.time()
        gen_batch = server.generate_sequences(batch)
        t_gen_end = time.time()
        server.sleep()
        t_end = time.time()

        wake_up_time = t_gen_start - t_start
        gen_time = t_gen_end - t_gen_start
        sleep_time = t_end - t_gen_end
        step_time = t_end - t_start
        print(f"[DEBUG] backend: {backend}, n_gpus_per_node: {n_gpus_per_node}, use_chat_scheduler: True, batch_size: {len(gen_batch)}, step: {step}, step time: {step_time:.2f} secs, wake_up_time: {wake_up_time:.2f} secs, gen_time: {gen_time:.2f} secs, sleep_time: {sleep_time:.2f} secs")
        break

    ray.shutdown()


def perf_sync_llm(backend, n_gpus_per_node):
    """Perf test sync LLM."""
    config = init_config(n_gpus_per_node)
    config.actor_rollout_ref.rollout.mode = "sync"
    actor_rollout_wg, server_address, dataloader = initialize(config, backend)

    for step, batch in enumerate(dataloader):
        batch: DataProto = DataProto.from_single_dict(batch)
        batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )
        t_start = time.time()
        gen_batch = actor_rollout_wg.generate_sequences(batch)
        t_end = time.time()
        print(f"[DEBUG] backend: {backend}, n_gpus_per_node: {n_gpus_per_node}, use_chat_scheduler: False, batch_size: {len(gen_batch)}, step: {step}, step_time: {t_end - t_start:.2f} secs")
        break

    ray.shutdown()


if __name__ == "__main__":
    # Perf test sync LLM
    perf_sync_llm(backend="sync", n_gpus_per_node=2)

    # Perf test AsyncLLM backend:
    # - cgraph: default RayDistributedExecutor with compiled graph
    # - external: ExternalRayDistributedExecutor with remote call
    asyncio.run(perf_without_chat_scheduler(backend="cgraph", n_gpus_per_node=2))
    asyncio.run(perf_without_chat_scheduler(backend="external", n_gpus_per_node=2))

    # Perf test ChatScheduler scalibility:
    # - n_gpus_per_node=2: 1 instance
    # - n_gpus_per_node=8: 4 instances
    perf_with_chat_scheduler(backend="external", n_gpus_per_node=2)
    perf_with_chat_scheduler(backend="external", n_gpus_per_node=8)
