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
import asyncio
import re
from typing import Any, Dict, List

import aiohttp
from openai.types.chat.chat_completion import ChatCompletion

from verl.protocol import DataProto
from verl.workers.rollout.async_server import ChatCompletionScheduler

ci_user_prompt_template_v3 = """Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. 
Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.
The last part of your response should be in the following format:
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>

*user question:*
{question}

Remember to place the final answer in the last part using the format: 
<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"""  # noqa: E501


class ToolChatCompletionScheduler(ChatCompletionScheduler):
    """This is a demo chat completion scheduler that supports sandbox code execution
    described in ReTool paper: https://arxiv.org/pdf/2504.11536
    """

    def __init__(self, config, model_path, server_addresses, sandbox_url, user_prompt_template=ci_user_prompt_template_v3, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.sandbox_url = sandbox_url
        self.user_prompt_template = user_prompt_template

    async def sandbox_code_execution(self, code: str) -> Dict[str, Any]:
        """Execute python code in sandbox."""
        try:
            session = aiohttp.ClientSession()
            async with session.post(
                url=self.sandbox_url,
                json={"code": code},
            ) as resp:
                return await resp.json()
        finally:
            await session.close()

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            include_stop_str_in_output=True,
            stop=["</answer>", "</code>"],
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        kwargs.update(sampling_params)
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}"
            batch_conversations, batch_index, turn = (
                info["batch_conversations"],
                info["batch_index"],
                info["turn"],
            )
            role, content, finish_reason = completions.choices[0].message.role, completions.choices[0].message.content, completions.choices[0].finish_reason
            batch_conversations[batch_index].append({"role": role, "content": content})

            # STEP 0: check if we reach max tokens
            if finish_reason == "length":
                print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Reach max tokens, done!")
                return

            # STEP 1: check if we got answer
            matches = re.findall(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if matches:
                print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Got answer: {matches[0]}, done!")
                return

            # STEP 2: check if we got code block
            matches = re.findall(r"<code>\s*```python(.*?)```\s*</code>", content, re.DOTALL)
            if not matches:
                print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] No code block found, done!")
                return

            # STEP 3: execute code block in sandbox
            code = matches[0].strip()
            result = await self.sandbox_code_execution(code)
            stdout, stderr = result["stdout"], result["stderr"]
            batch_conversations[batch_index].append({"role": "tool", "content": f"<interpreter>{stdout}{stderr}</interpreter>"})
            print(f"[id={completions.id},turn={turn},finish_reason={finish_reason}] Code block executed, continue...")

            # STEP 4: resubmit chat completions with code block output
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "turn": turn + 1,
                },
                model=self.model_name,
                messages=batch_conversations[batch_index],
                extra_headers=extra_headers,
                **kwargs,
            )

        tasks, batch_conversations = [], [None] * len(batch)
        for batch_index, conversation in enumerate(batch.non_tensor_batch["raw_prompt"]):
            # raw_prompt: [{"role": "user", "content": ""}, ["role": "assistant", "content"], ...]
            conversation[0]["content"] = self.user_prompt_template.replace("{question}", conversation[0]["content"])
            batch_conversations[batch_index] = conversation.tolist()

            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "turn": 1,
                        },
                        model=self.model_name,
                        messages=batch_conversations[batch_index],
                        **kwargs,
                    )
                )
            )

        await asyncio.gather(*tasks)
        print("[ToolChatCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, batch_conversations)

    def _postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]]) -> DataProto:
        # TODO: implement loss mask
        return batch_conversations
