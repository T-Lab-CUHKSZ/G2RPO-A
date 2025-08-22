# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]

##############################################################################
from verl.trainer.ppo.ray_trainer import Role
from collections import deque
#############################################################################

def debug_complete_batch_structure(self, old_prompts, mixed_prompts, batch, guided_idx, noguid_idx, 
                                   n=8, k=2, P=None, G=None):
    """
    验证整个批次的结构：
    1. 每个原始prompt被复制n次
    2. 每组n个样本中，前k个有引导，后(n-k)个无引导
    3. UID分组正确
    4. 生成的回复结构正确
    """
    
    print("\n" + "="*120)
    print("🔍 完整批次结构验证")
    print("="*120)
    
    B = old_prompts.shape[0]  # 原始批次大小
    total_samples = mixed_prompts.shape[0]  # 总样本数
    
    print(f"📊 批次基本信息:")
    print(f"  原始批次大小 B: {B}")
    print(f"  复制倍数 n: {n}")
    print(f"  有引导数 k: {k}")
    print(f"  总样本数: {total_samples} (期望: {B * n})")
    print(f"  有引导样本数: {len(guided_idx)} (期望: {B * k})")
    print(f"  无引导样本数: {len(noguid_idx)} (期望: {B * (n-k)})")
    
    # 基本数量验证
    assert total_samples == B * n, f"总样本数不匹配: {total_samples} != {B * n}"
    assert len(guided_idx) == B * k, f"有引导样本数不匹配: {len(guided_idx)} != {B * k}"
    assert len(noguid_idx) == B * (n-k), f"无引导样本数不匹配: {len(noguid_idx)} != {B * (n-k)}"
    
    print(f"✅ 基本数量验证通过")
    
    # 验证每个原始样本的8个副本
    print(f"\n📋 详细验证每个原始样本的 {n} 个副本:")
    
    for original_idx in range(B):
        print(f"\n--- 原始样本 {original_idx} ---")
        
        # 获取原始prompt
        original_prompt_ids = old_prompts[original_idx]
        original_prompt_text = self.tokenizer.decode(original_prompt_ids, skip_special_tokens=True)
        print(f"  原始Prompt: {repr(original_prompt_text[-1000:])}...")
        
        # 获取这个原始样本对应的n个副本的索引
        replica_indices = list(range(original_idx * n, (original_idx + 1) * n))
        print(f"  副本索引: {replica_indices}")
        
        # 验证每个副本
        guided_count = 0
        noguid_count = 0
        
        for replica_idx, global_idx in enumerate(replica_indices):
            is_guided = global_idx in guided_idx
            is_noguid = global_idx in noguid_idx
            
            # 获取这个副本的完整prompt
            if P is not None and G is not None:
                replica_full_prompt = mixed_prompts[global_idx, :P+G]
                replica_prompt_part = mixed_prompts[global_idx, :P]
                replica_guidance_part = mixed_prompts[global_idx, P:P+G]
                
                replica_full_text = self.tokenizer.decode(replica_full_prompt, skip_special_tokens=True)
                replica_prompt_text = self.tokenizer.decode(replica_prompt_part, skip_special_tokens=True)
                replica_guidance_text = self.tokenizer.decode(replica_guidance_part, skip_special_tokens=True)
            else:
                replica_full_prompt = mixed_prompts[global_idx]
                replica_full_text = self.tokenizer.decode(replica_full_prompt, skip_special_tokens=True)
                replica_prompt_text = replica_full_text
                replica_guidance_text = ""
            
            # 期望：前k个应该有引导，后(n-k)个应该无引导
            expected_guided = replica_idx < k
            
            status = "✅" if (is_guided == expected_guided) else "❌"
            guidance_status = "有引导" if is_guided else "无引导"
            
            print(f"    副本 {replica_idx} (全局索引 {global_idx}): {status} {guidance_status}")
            
            if is_guided:
                guided_count += 1
                print(f"      完整文本: {repr(replica_full_text[-1500:])}...")
                if G and G > 0:
                    print(f"      引导部分: {repr(replica_guidance_text[-500:])}...")
            else:
                noguid_count += 1
                print(f"      Prompt文本: {repr(replica_prompt_text[-1000:])}...")
            
            # 验证prompt部分是否与原始一致
            prompt_matches = replica_prompt_text.strip() == original_prompt_text.strip()
            if not prompt_matches:
                print(f"      ❌ Prompt不匹配原始!")
                print(f"         原始: {repr(original_prompt_text[:500])}...")
                print(f"         副本: {repr(replica_prompt_text[:500])}...")
        
        # 验证这个组内的引导分布
        expected_guided_count = k
        expected_noguid_count = n - k
        
        print(f"  组内统计: 有引导 {guided_count}/{expected_guided_count}, 无引导 {noguid_count}/{expected_noguid_count}")
        
        if guided_count != expected_guided_count or noguid_count != expected_noguid_count:
            print(f"  ❌ 组内分布错误!")
        else:
            print(f"  ✅ 组内分布正确")
    
    # 验证UID分组（如果存在）
    print(f"\n🆔 UID分组验证:")
    if hasattr(batch, 'non_tensor_batch') and 'uid' in batch.non_tensor_batch:
        uids = batch.non_tensor_batch['uid']
        print(f"  UID数组长度: {len(uids)}")
        
        unique_uids, counts = np.unique(uids, return_counts=True)
        print(f"  唯一UID数: {len(unique_uids)} (期望: {B})")
        print(f"  每个UID的计数: {counts} (期望: 全部为 {n})")
        
        uid_correct = len(unique_uids) == B and all(c == n for c in counts)
        print(f"  UID分组: {'✅ 正确' if uid_correct else '❌ 错误'}")
        
        # 详细检查每个UID组
        for i, (uid, count) in enumerate(zip(unique_uids[:3], counts[:3])):  # 只检查前3个
            print(f"    UID {i}: {uid} -> {count} 个样本")
            uid_indices = np.where(uids == uid)[0]
            print(f"      对应索引: {uid_indices.tolist()}")
    else:
        print(f"  ❌ 没有找到UID信息")
    
    # 验证生成的回复结构（如果存在）
    print(f"\n💬 生成回复验证:")
    if hasattr(batch, 'batch') and 'responses' in batch.batch:
        responses = batch.batch['responses']
        print(f"  回复张量形状: {responses.shape}")
        print(f"  期望形状: ({B * n}, 回复长度)")
        
        # 检查每组的回复
        for original_idx in range(min(3, B)):  # 只检查前3个原始样本
            print(f"\n  --- 原始样本 {original_idx} 的回复 ---")
            replica_indices = list(range(original_idx * n, (original_idx + 1) * n))
            
            for replica_idx, global_idx in enumerate(replica_indices):
                response = responses[global_idx]
                response_text = self.tokenizer.decode(response, skip_special_tokens=True)
                
                is_guided = global_idx in guided_idx
                guidance_status = "有引导" if is_guided else "无引导"
                
                print(f"    副本 {replica_idx} ({guidance_status}): {repr(response_text[:1000])}...")
                
                # 如果有引导，检查回复前面是否包含引导内容
                if is_guided and G and G > 0:
                    guidance_part_in_response = response[:G]
                    guidance_in_response_text = self.tokenizer.decode(guidance_part_in_response, skip_special_tokens=True)
                    print(f"      回复中的引导: {repr(guidance_in_response_text[:500])}...")
    else:
        print(f"  ❌ 没有找到回复信息")
    
    # 最终验证总结
    print(f"\n📝 验证总结:")
    print(f"  ✅ 批次大小正确: {B} -> {B * n}")
    print(f"  ✅ 引导分布正确: {len(guided_idx)} 有引导, {len(noguid_idx)} 无引导")
    print(f"  ✅ 每组结构正确: 前 {k} 个有引导, 后 {n-k} 个无引导")
    
    print("="*120 + "\n")

def quick_batch_check(self, old_prompts, mixed_prompts, batch, guided_idx, noguid_idx, n=8, k=2):
    """快速检查批次结构是否正确"""
    
    B = old_prompts.shape[0]
    
    print(f"\n🔍 快速批次检查:")
    print(f"  原始样本: {B}, 总样本: {mixed_prompts.shape[0]} (期望: {B*n})")
    print(f"  有引导: {len(guided_idx)} (期望: {B*k}), 无引导: {len(noguid_idx)} (期望: {B*(n-k)})")
    
    # 检查每组的前几个样本
    print(f"\n📋 每组结构检查 (只显示前2组):")
    for group in range(min(2, B)):
        print(f"\n  组 {group} (索引 {group*n} 到 {(group+1)*n-1}):")
        
        # 获取原始prompt
        original_text = self.tokenizer.decode(old_prompts[group], skip_special_tokens=True)
        print(f"    原始: {repr(original_text[:800])}...")
        
        # 检查这组的每个副本
        for i in range(n):
            global_idx = group * n + i
            is_guided = global_idx in guided_idx
            expected_guided = i < k
            
            # 获取这个副本的prompt
            replica_prompt = mixed_prompts[global_idx]
            replica_text = self.tokenizer.decode(replica_prompt, skip_special_tokens=True)
            
            status = "✅" if (is_guided == expected_guided) else "❌"
            guidance_str = "有引导" if is_guided else "无引导"
            
            print(f"    副本{i}: {status} {guidance_str} | {repr(replica_text[:1000])}...")
    
    # 检查UID分组
    if hasattr(batch, 'non_tensor_batch') and 'uid' in batch.non_tensor_batch:
        unique_uids, counts = np.unique(batch.non_tensor_batch['uid'], return_counts=True)
        uid_ok = len(unique_uids) == B and all(c == n for c in counts)
        print(f"\n🆔 UID分组: {'✅ 正确' if uid_ok else '❌ 错误'} ({len(unique_uids)} 个UID, 每个 {counts[0] if len(counts) > 0 else 0} 个样本)")
    
    # 检查回复（如果存在）
    if hasattr(batch, 'batch') and 'responses' in batch.batch:
        responses = batch.batch['responses']
        print(f"\n💬 回复: 形状 {responses.shape}, 期望 ({B*n}, 回复长度)")
        
        # 显示第一组的回复结构
        if B > 0:
            print(f"  第一组回复预览:")
            for i in range(min(n, 4)):  # 最多显示4个
                global_idx = i
                response_text = self.tokenizer.decode(responses[global_idx], skip_special_tokens=True)
                is_guided = global_idx in guided_idx
                guidance_str = "有引导" if is_guided else "无引导"
                print(f"    副本{i} ({guidance_str}): {repr(response_text[:800])}...")
    
    print()

class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class AdaptiveGRPORayTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    #####################################################################################################
    # —— 1. 可选超参（也可以直接从 config.algorithm 里读） ——  
    INITIAL_GUIDANCE_LEN = 3072
    FINAL_GUIDANCE_LEN = 0
    MAX_GUIDANCE_LEN = 3072
    #############################################################################################

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # 改为固定引导长度
        self._init_guidance_scheduler()

    def _init_guidance_scheduler(self):
        """初始化引导长度调度器"""
        self.initial_guidance_length = self.INITIAL_GUIDANCE_LEN
        self.final_guidance_length = self.FINAL_GUIDANCE_LEN
        
        # 计算每步的递减量
        total_steps = self.total_training_steps
        if total_steps <= 1:
            self.guidance_step_reduction = 0
            self.guidance_length = self.final_guidance_length
        else:
            # 线性递减：从 initial 到 final，总共 total_steps 步
            self.guidance_step_reduction = (self.initial_guidance_length - self.final_guidance_length) / (total_steps - 1)
            self.guidance_length = self.initial_guidance_length
        
        print(f"引导长度调度器初始化:")
        print(f"  初始长度: {self.initial_guidance_length}")
        print(f"  最终长度: {self.final_guidance_length}")
        print(f"  总训练步数: {total_steps}")
        print(f"  每步递减: {self.guidance_step_reduction:.2f}")
        print(f"  第1步引导长度: {self.guidance_length}")

    def _update_guidance_length(self):
        """根据当前步数更新引导长度"""
        if self.guidance_step_reduction == 0:
            return  # 不需要调整
        
        # 线性插值计算当前步的引导长度
        # global_steps 从 1 开始，所以需要减 1
        progress = (self.global_steps - 1) / (self.total_training_steps - 1)
        progress = max(0.0, min(1.0, progress))  # 确保在 [0, 1] 范围内
        
        self.guidance_length = int(
            self.initial_guidance_length - 
            progress * (self.initial_guidance_length - self.final_guidance_length)
        )
        
        # 确保不会小于最终长度
        self.guidance_length = max(self.final_guidance_length, self.guidance_length)
        
        print(f"步骤 {self.global_steps}: 引导长度更新为 {self.guidance_length} (进度: {progress:.2%})")

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        if hasattr(batch, "meta_info"):
            gt = batch.meta_info.get("global_token_num", None)
            print(f"[PRINT] 进入 _balance_batch 前，batch.meta_info['global_token_num'] = {gt}", flush=True)
            print(f"[PRINT] 进入 _balance_batch 前，len(batch) = {len(batch)}", flush=True)
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:

                self._update_guidance_length()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "generations"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                ######################################################################################
                old_prompts      = gen_batch.batch["input_ids"]
                generations_text = gen_batch.non_tensor_batch["generations"]
                gen_texts_list = generations_text.tolist()
                gen_ids = self.tokenizer(
                    gen_texts_list,
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.MAX_GUIDANCE_LEN,
                    return_tensors="pt",
                ).input_ids.to(old_prompts.device)

                guidance_fixed = []
                for ids in gen_ids:
                    if self.guidance_length == 0:
                        # 当 guidance_length = 0 时，创建一个空的 long 类型张量
                        ids = torch.empty(0, dtype=old_prompts.dtype, device=ids.device)
                    elif ids.numel() >= self.guidance_length:
                        ids = ids[: self.guidance_length]
                    else: # 如果generations里面的内容比目标截取长度小，就pad了之后再截取
                        pad = torch.full(
                            (self.guidance_length - ids.numel(),),
                            self.tokenizer.pad_token_id,
                            device=ids.device,
                            dtype=ids.dtype,
                        )
                        ids = torch.cat([ids, pad], dim=0)
                    guidance_fixed.append(ids)
                guidance_ids = torch.stack(guidance_fixed, dim=0)
                # 因为原verl逻辑是在generate_sequence里面去进行G次回复，而不是先复制G次prompt再统一生成，因此我们不得不将原逻辑的n永远设置为1，然后自己在trainer里面复制n次再传递给generate_sequence
                n = 8     # 例如 8
                k = 0     # 例如 2
                assert 0 <= k <= n, "k_guided 必须 0≤k≤n"

                pad_id = self.tokenizer.pad_token_id

                # ==== 1. repeat 原始 prompt n 倍 ====
                B, P = old_prompts.size() # B 和 P 分别是 batch size 和原始 prompt 的长度
                prompt_rep = old_prompts.repeat_interleave(n, dim=0)           # 这是repeat的原来的prompt(B*n, P)

                # ==== 2. 生成 guidance_rows 布尔索引 ====
                # 每组 n 行：前 k 行 True → 带引导；后 n-k 行 False → 不带引导
                pattern = torch.arange(n, device=old_prompts.device) < k # torch.arange(n) < k==tensor([True, True, False, False, False, False, False, False]) 然后将其扩展为n行
                guidance_rows_bool = pattern.repeat(B)                      # (B*n,)，确保真的是布尔值

                guided_idx  = guidance_rows_bool.nonzero(as_tuple=True)[0]     # 这里提取的是有引导的prompt的索引(B*k,)
                noguid_idx  = (~guidance_rows_bool).nonzero(as_tuple=True)[0]  # ~是取反的意思(B*(n-k),)

                # ==== 3. 扩展 guidance_ids 到 (B*k, G) ====
                G = self.guidance_length
                guidance_ids_big = guidance_ids.repeat_interleave(k, dim=0)    # 上面repeat的是prompt的索引，这里repeat的是引导本身(B*k, G)

                # ==== 4. 构造 mixed_prompts / attention_mask / position_ids ====
                seq_len = P + G
                mixed_prompts = torch.full((B*n, seq_len),
                                        pad_id,
                                        device=old_prompts.device,
                                        dtype=old_prompts.dtype) # 创建一个全为pad的，形如 (B*n, P+G) 的张量

                # 4.1 不带引导行：只放原 prompt
                mixed_prompts[noguid_idx, :P] = prompt_rep[noguid_idx] # 找到索引为没有引导的行然后再对准前面P个token，也就是原prompt的长度，放置复制好的原prompt

                # 4.2 带引导行：prompt | guidance
                mixed_prompts[guided_idx] = torch.cat(
                    [prompt_rep[guided_idx], guidance_ids_big], dim=1 # 注意这里是把有引导的prompt取出来单独进行拼接，而不是在原prompt组里面进行拼接，因此guidance_ids_big不需要知道它在原来的prompt组中的位置
                )                                            # 找到索引为有引导的行，其全部列都要被填充，因此不用再索引一次，然后直接把复制好的有引导的原prompt与引导拼接后怼进去(B*k, P+G)

                # 4.3 attention_mask: 非 pad → 1
                mixed_attention_mask = (mixed_prompts != pad_id).long()        # 如果该位置不是pad_token，那么就为1，是pad_token，就为0(B*n, P+G)

                # 4.4 position_ids: 每行 0..P+G-1
                mixed_position_ids = torch.arange(seq_len, device=old_prompts.device).unsqueeze(0).repeat(B*n, 1)              # 初始化position_ids (B*n, P+G)

                gen_batch = gen_batch.repeat(repeat_times=n, interleave=True) # 很难想象之前竟然没有repeat gen_batch
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object) # 这个分组必须在repeat前面
                batch = batch.repeat(repeat_times=n, interleave=True) # batch肯定也要重复，不然咋拼接

                # 写回 gen_batch
                gen_batch.batch["input_ids"]        = mixed_prompts
                gen_batch.batch["attention_mask"] = mixed_attention_mask
                gen_batch.batch["position_ids"]   = mixed_position_ids

                quick_batch_check(self, old_prompts, mixed_prompts, batch, guided_idx, noguid_idx, n=n, k=k)

                ############################################################################################################################################################

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            ########################################################################################################################
                            # ==== 5. 调 vLLM 生成 ====
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            responses = gen_batch_output.batch["responses"]                # (B*n, R)

                            # ==== 6. 给带引导行把 guidance 再拼到 response 前 ====
                            responses_guided = torch.cat([guidance_ids_big, responses[guided_idx]], dim=1) # 将有引导的索引的回复取出来然后与引导进行拼接
                            # 给无引导行前面补 G 个 pad，让所有行等长
                            pad_for_noguid   = torch.full((noguid_idx.size(0), G),
                                                        pad_id,
                                                        device=responses.device,
                                                        dtype=responses.dtype)
                            responses_noguid = torch.cat([pad_for_noguid, responses[noguid_idx]], dim=1) # 将无引导的索引的回复取出来然后与pad进行拼接

                            merged_resp = torch.empty((B*n, G + responses.shape[1]),
                                                    device=responses.device,
                                                    dtype=responses.dtype) # 整个responses的形状是(B*n, G+R)，其中G是引导长度，R是回复长度，这里是一个初始化
                            merged_resp[guided_idx] = responses_guided # 将拼接好的有引导的回复放到对应的行
                            merged_resp[noguid_idx] = responses_noguid # 将拼接好的无引导的回复放到对应的行
                            gen_batch_output.batch["responses"] = merged_resp               # (B*n, G+R) 更新batch里面的response

                            # ==== 7. 还原 prompts / mask / pos_ids 为纯 prompt (B*n, P) ====
                            # pure_prompts = old_prompts.repeat_interleave(n, dim=0)          # (B*n, P)
                            # gen_batch.batch["input_ids"] = pure_prompts # 从此之后，prompts 就是纯 prompt 了，不再包含引导部分
                            # gen_batch.batch["attention_mask"] = torch.ones((B*n, P), dtype=torch.long, device=old_prompts.device)
                            # gen_batch.batch["position_ids"] = torch.arange(P, device=old_prompts.device)\
                            #                                 .unsqueeze(0).repeat(B*n, 1) position_ids和attention_mask都不需要还原，因为它们是对整个序列的。

                            # 其余 reward / log_prob / advantage 逻辑保持不变
                            batch = batch.union(gen_batch_output)
                            # debug_complete_batch_structure(self, old_prompts, mixed_prompts, batch, guided_idx, noguid_idx, n=n, k=k, P=P, G=G)
                            ########################################################################################################################
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    ###############################################################################################################
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    #############################################################################################################

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch: # 默认为True，详见verl/verl/trainer/config/ppo_trainer.yaml
                        ##################################################################################
                        if hasattr(batch, "meta_info"):
                            # 打印最关心的 global_token_num
                            global_tokens = batch.meta_info.get("global_token_num", None)
                            print(f"[DEBUG] repeat 之后，batch.meta_info['global_token_num'] = {global_tokens}", flush=True)
                            # 也可以顺便看一下 batch 的逻辑长度（DataProto.__len__）
                            print(f"[DEBUG] repeat 之后，len(batch) = {len(batch)}", flush=True)
                        else:
                            print("[DEBUG] repeat 之后，batch 没有 meta_info 属性！", flush=True)
                        ############################################################################################
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            # 读取accuracy_reward并且保存下来以供后续使用
                            if reward_extra_infos_dict is not None and "score" in reward_extra_infos_dict: # 再在dapo.py里面确保一下返回的是“score”还是“reward_tensor”。
                                original_scores = reward_extra_infos_dict["score"]
                                fixed_scores = [0.0 if score == -1 else score for score in original_scores]
                                metrics["accuracy_reward"] = fixed_scores

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        print("\n" + "="*80)
                        print("即将调用 compute_log_prob，当前 batch 中各张量的形状:")
                        
                        # 获取各部分的形状信息
                        input_ids_shape = batch.batch.get("input_ids", "不存在").shape
                        attention_mask_shape = batch.batch.get("attention_mask", "不存在").shape
                        responses_shape = batch.batch.get("responses", "不存在").shape
                        position_ids_shape = batch.batch.get("position_ids", "不存在").shape
                        
                        # 打印形状
                        print(f"  - input_ids.shape     : {input_ids_shape}")
                        print(f"  - attention_mask.shape: {attention_mask_shape}")
                        print(f"  - responses.shape     : {responses_shape}")
                        print(f"  - position_ids.shape  : {position_ids_shape}")
                        
                        # 结合我们之前的讨论进行解释
                        print("\n解释 (P=prompt长度, G=引导长度, R=响应长度):")
                        print(f"  - input_ids 的长度应为 P+G+R: {input_ids_shape[1]}")
                        print(f"  - responses 的长度应为 G+R: {responses_shape[1]}")
                        print(f"  - position_ids 的长度应为 P+G+R: {position_ids_shape[1]}")
                        print("="*80 + "\n")
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch) # 这个是当前策略下的log‐prob，可以看到它是直接把batch传入进行计算的，那么由于此时batch里面的responses已经包含了前面的引导部分，所以这里的old_log_prob也是包含了前面的引导部分的。
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        # if "rollout_log_probs" in batch.batch.keys():
                        #     # TODO: we may want to add diff of probs too.
                        #     rollout_old_log_probs = batch.batch["rollout_log_probs"] # 这里的rollout_log_probs是没有包含引导部分的，而且根本没有任何方式可以得到引导的lp，反正这一段不参与loss计算，只是用于监控，我直接不要了！
                        #     actor_old_log_probs = batch.batch["old_log_probs"]
                        #     attention_mask = batch.batch["attention_mask"]
                        #     responses = batch.batch["responses"]
                        #     response_length = responses.size(1)
                        #     response_mask = attention_mask[:, -response_length:]

                        #     rollout_probs = torch.exp(rollout_old_log_probs)
                        #     actor_probs = torch.exp(actor_old_log_probs)
                        #     rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                        #     rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                        #     rollout_probs_diff_max = torch.max(rollout_probs_diff)
                        #     rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                        #     rollout_probs_diff_std = torch.std(rollout_probs_diff)
                        #     metrics.update(
                        #         {
                        #             "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        #             "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        #             "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                        #         }
                        #     )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                    
                    # 把 guidance_length 写入 metrics，结合父类的其他 metric 一并打日志
                    metrics["training/guidance_length"] = self.guidance_length
                    logger.log(data=metrics, step=self.global_steps)
                    #############################################################################################################################

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                metrics["training/guidance_length"] = self.guidance_length
                metrics["training/guidance_progress"] = (self.global_steps - 1) / (self.total_training_steps - 1)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
