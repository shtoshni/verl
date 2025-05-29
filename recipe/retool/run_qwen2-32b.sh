set -x

# ================= data/model/tool =================
DATA_ROOT=

dapo_math_17k=${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k
aime_2024=${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024
model_path=${DATA_ROOT}/JoeYing/ReTool-Qwen-32B-SFT

train_files="['$dapo_math_17k']"
test_files="['$aime_2024']"

chat_scheduler=recipe.retool.chat_scheduler.ToolChatCompletionScheduler
sandbox_fusion_url="https://***.apigateway-cn-beijing.volceapi.com/run_code"

# ================= perfomance =================
ARNOLD_WORKER_GPU=16
ARNOLD_WORKER_NUM=8

TP=4 # vllm
SP=4 # train

offload=True

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=examples/ppo_trainer/runtime_env.yaml \
    --entrypoint-num-cpus=1 \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=1024 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=recipe/retool/chat_scheduler.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/retool/chat_scheduler.py \
    custom_reward_function.name=compute_score \
    reward_model.sandbox_fusion.url=$sandbox_fusion_url \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.chat_scheduler=$chat_scheduler \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=98304 \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='wuxibin_verl_example_gsm8k' \
    trainer.experiment_name='qwen2-32b_retool'-$(date +"%Y-%m-%d-%H-%M-%S") \
    trainer.n_gpus_per_node=$ARNOLD_WORKER_GPU \
    trainer.val_before_train=True \
    trainer.log_val_generations=100 \
    trainer.nnodes=$ARNOLD_WORKER_NUM \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
