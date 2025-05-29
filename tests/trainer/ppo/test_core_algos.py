# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_gae_advantage_return, verl_F


@pytest.fixture
def train_batch():
    token_level_rewards = torch.tensor(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.bfloat16,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
        ]
    )
    values = torch.rand_like(token_level_rewards)

    batch = DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": token_level_rewards,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "values": values,
            },
            batch_size=len(token_level_rewards),
        ),
        meta_info={
            "gamma": 0.95,
            "lam": 0.95,
        },
    )

    return batch


def test_compute_gae_advantage_return_single_turn(train_batch):
    token_level_rewards = train_batch.batch["token_level_rewards"]
    response_mask = train_batch.batch["attention_mask"]
    values = train_batch.batch["values"]
    gamma = train_batch.meta_info["gamma"]
    lam = train_batch.meta_info["lam"]

    expected_advantages, expected_returns = compute_gae_advantage_return(
        token_level_rewards,
        values,
        response_mask,
        gamma,
        lam,
    )

    advantages = torch.zeros_like(token_level_rewards)
    for i in range(advantages.shape[0]):
        nextvalues, lastgaelam = 0, 0
        for j in reversed(range(advantages.shape[1])):
            delta = token_level_rewards[i][j] + gamma * nextvalues - values[i][j]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages[i][j] = lastgaelam
            nextvalues = values[i][j]

    returns = advantages + values
    advantages = verl_F.masked_whiten(advantages, response_mask)

    torch.testing.assert_close(expected_advantages, advantages)
    torch.testing.assert_close(expected_returns, returns)


def test_compute_gae_advantage_return_multi_turn(train_batch):
    token_level_rewards = train_batch.batch["token_level_rewards"]
    response_mask = train_batch.batch["response_mask"]
    values = train_batch.batch["values"]
    gamma = train_batch.meta_info["gamma"]
    lam = train_batch.meta_info["lam"]

    expected_advantages, expected_returns = compute_gae_advantage_return(
        token_level_rewards,
        values,
        response_mask,
        gamma,
        lam,
        multi_turn=True,
    )

    advantages = torch.zeros_like(token_level_rewards)
    for i in range(advantages.shape[0]):
        nextvalues, lastgaelam = 0, 0
        for j in reversed(range(advantages.shape[1])):
            delta = token_level_rewards[i][j] + gamma * nextvalues - values[i][j]
            lastgaelam = delta + gamma * lam * lastgaelam

            nextvalues = values[i][j] if response_mask[i][j] else nextvalues
            lastgaelam = lastgaelam if response_mask[i][j] else (advantages[i][j + 1] if j < advantages.shape[1] - 1 else 0)

            advantages[i][j] = lastgaelam

    returns = advantages + values
    advantages = verl_F.masked_whiten(advantages, response_mask)
    print(f"returns: {returns}")
    print(f"returns: {advantages}")

    torch.testing.assert_close(expected_advantages, advantages)
    torch.testing.assert_close(expected_returns, returns)
