# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Iterator
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

import math
from transformers import TrainerCallback

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
#####################################################################################################################################################
class GuidanceLengthUpdateCallback(TrainerCallback): # 这个类继承自transformers.TrainerCallback，会在每个全局训练步骤执行完的时候自动调用
    """在每个训练步骤结束后更新 guidance_length"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self._acc_sum = 0.0
        self._acc_cnt = 0      

    def on_step_end(self, args, state, control, **kwargs):
        """在每个训练步骤结束时调用"""
        # 收集当前步骤累积的所有准确率
        if hasattr(self.trainer, '_step_accuracies') and self.trainer._step_accuracies: # hasattr(self.trainer, '_step_accuracies')是在检查trainer中是否有_step_accuracies这个属性，self.trainer._step_accuracies是在检查这个属性是否有值
            all_accuracies = self.trainer._step_accuracies.copy() # _step_accuracies是一个列表，其中一个元素是一个micro-step的accuracy_reward。如果我把梯度累积设置为4，那么每一个global step的_step_accuracies就会有四个元素
            avg_accuracy = sum(all_accuracies) / len(all_accuracies) # 因此这个求出来的就是global step的总的accuracy_reward
            
            if self.trainer.accelerator.is_main_process: # 每一个global step只打印一次这个信息
                print(f"Step {state.global_step}: Collected {len(all_accuracies)} micro-batch accuracies: {all_accuracies}")
                print(f"Step {state.global_step}: Average accuracy for this step: {avg_accuracy:.4f}")
        else:
            avg_accuracy = 0.0 # 如果没发现这个属性那么就创一个并且将初值赋为0。
            if self.trainer.accelerator.is_main_process:
                print(f"Step {state.global_step}: No accuracies collected, using 0.0")
        
        # 在真正的训练步骤结束时进行更新
        if state.global_step > 0:
            # 主进程计算新的 guidance_length
            if self.trainer.accelerator.is_main_process:
                old_guidance_length = self.trainer.guidance_length
                new_guidance_length = self.trainer._compute_new_guidance_length(avg_accuracy, state.global_step)
                
                # 只有当真正发生更新时才打印更新信息
                if new_guidance_length != old_guidance_length: 
                    print(f"Step {state.global_step}: Updating guidance_length {old_guidance_length} -> {new_guidance_length}")
                else: # 不光accuracy_reward完全没变时guidance_length会保持不变，更常见的情况是guidance_length达到了最大值后如果还需要增加，那么这种情况也不会发生改变。
                    print(f"Step {state.global_step}: guidance_length unchanged at {old_guidance_length}")
            else:
                new_guidance_length = None
            
            # 同步新的 guidance_length 到所有进程
            new_len_list = [new_guidance_length]
            broadcast_object_list(new_len_list, from_process=0)
            self.trainer.guidance_length = new_len_list[0]
            
            # 重置累积器和步骤级别的变量
            self.trainer._step_accuracies = [] # 注意当前函数至始至终没有说_step_accuracies是如何赋值的，那是因为该属性并不是在这个函数里面被计算的，详见_prepare_inputs函数
            self.trainer._step_guidance_length = None
            
            # 只在主进程输出同步确认
            if self.trainer.accelerator.is_main_process:
                print(f"Step {state.global_step}: guidance_length synced to all processes")


########################################################################按顺序遍历############################################################################
class RepeatSequentialSampler(Sampler[int]):
    """每个样本连续重复 num_repeats 次，按顺序遍历。"""

    def __init__(self, data_source, num_repeats: int):
        self.data_source = data_source
        self.num_repeats = num_repeats
        self.num_samples = len(data_source)

    def __iter__(self) -> Iterator[int]:
        # 每个 index 连续重复 num_repeats 次
        return iter(
            idx
            for idx in range(self.num_samples)
            for _ in range(self.num_repeats)
        )

    def __len__(self) -> int:
        return self.num_samples * self.num_repeats


############################################################################################################################################################

class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
                    None, None),
            peft_config: Optional["PeftConfig"] = None,
            _prev_accuracy_rewards=None,  # type: Optional[torch.Tensor]
            guidance_length=None,  # type: Optional[torch.Tensor]
            max_guidance_length=3072,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        # (发生): so, "model" has two possibilities: 1) it is a string. 2) it is an instantiation of "PreTrainedModel".
        # If it is a string, showing as follows, we need to "from_pretrained" it.
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        # If it is already an instantiation of "PreTrainedModel", then, we could directly load it instead of re-"from_pretrained" it.
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif not is_peft_model(model):
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            # PEFT: Parameter-Efecient Fine-Tuning, e.g., LoRA.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        # This is simply the initation.
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        # This part also indicates that it is allowed for the reward model to not to be a LLM.
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm

        self.beta = args.beta
        # self.beta = 0

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        ######################################################把新增变量扩展到全局########################################################
        self._prev_accuracy_rewards = []
        self.guidance_length = guidance_length
        self.max_guidance_length = max_guidance_length

        # 添加新的变量用于步骤管理
        self._step_accuracies = []         # 存储当前 global_step 的所有 micro-batch 准确率
        self._step_guidance_length = None   # 存储当前步骤使用的固定 guidance_length
        self._last_step = -1               # 记录上一次处理的步骤
        ###########################################################################################################################

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        #############################################################################################################################################
        # 添加自定义回调
        self.add_callback(GuidanceLengthUpdateCallback(self)) # 实例化自动回调的类。add_callback是Trainer的一个方法，它的主要作用是 允许你向训练器（Trainer）注册一个或多个回调（callback）对象
        #####################################################################################################################################

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    stop=["</answer>", "<|endoftext|>", "</ answer>"],
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False  # This variable is not directly used in this code. It is used in the father class of this class: "Tranier".

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()
    ################################################################################################################################################################
    def _compute_new_guidance_length(self, current_step_accuracy, global_step): # 这个函数也是只在一个global step内被调用一次，而不是micro step
        """
        以 3 步为一个周期：
          - 第 0 步 (mod 3 == 0)：收集 prev[0]，不更新
          - 第 1 步 (mod 3 == 1)：收集 prev[1]，不更新
          - 第 2 步 (mod 3 == 2)：用 (prev[0]+prev[1])/2 与当前步比较，
                                   得到新的 guidance_length，
                                   更新后立即把缓存重置，只留下本步 accuracy
                                   供下一循环使用
        更新完的 guidance_length 会在 *下一* 个 global_step
        进入 `_prepare_inputs` 时被锁定使用。
        """

        # 保证缓存是列表
        if not isinstance(self._prev_accuracy_rewards, list):
            self._prev_accuracy_rewards = []

        cycle_pos = (global_step - 1) % 3 # 这里之所以要减一是因为实践中发现会有一个初始值0，不减的话第一步的值会直接除以2作为两步的平均值

        # --- 前两步：只收集 ---
        if cycle_pos in (0, 1):
            if cycle_pos == 0:
                # 周期首步，清空缓存
                self._prev_accuracy_rewards = [current_step_accuracy]
            else:
                # 周期第二步，追加
                self._prev_accuracy_rewards.append(current_step_accuracy)

            if self.accelerator.is_main_process:
                print(
                    f"Step {global_step}: collected accuracy "
                    f"(cycle_pos={cycle_pos}) -> buffer={self._prev_accuracy_rewards}"
                )
            return self.guidance_length  # 不更新

        # --- 第三步：开始比较并决定 ---
        # 正常情况下这里一定已经有两条历史值
        if len(self._prev_accuracy_rewards) < 2:
            # 意外情况兜底，因为一般来说，前面流程走完之后_prev_accuracy_rewards里面已经有两个值了
            if self.accelerator.is_main_process:
                print(f"Step {global_step}: buffer size < 2, skip update")
            self._prev_accuracy_rewards = [current_step_accuracy]
            return self.guidance_length

        avg_prev_two = sum(self._prev_accuracy_rewards) / 2.0
        if self.accelerator.is_main_process:
            print(
                f"Step {global_step}: Comparison - "
                f"avg_prev_two_steps={avg_prev_two:.4f} vs current={current_step_accuracy:.4f}"
            )

        new_guidance_length = self.guidance_length  # 默认保持不变，好像有没有这句话都不影响
        if current_step_accuracy > 0 and avg_prev_two > 0:
            if current_step_accuracy < avg_prev_two:
                ratio = avg_prev_two / current_step_accuracy
                new_guidance_length = int(
                    min(self.guidance_length * ratio, self.max_guidance_length)
                )
                if self.accelerator.is_main_process:
                    print(
                        f"Step {global_step}: ↓performance, ↑guidance "
                        f"(ratio={ratio:.2f})"
                    )
            elif current_step_accuracy > avg_prev_two:
                ratio = avg_prev_two / current_step_accuracy
                new_guidance_length = int(
                    max(int(self.guidance_length * ratio), 1)
                )
                if self.accelerator.is_main_process:
                    print(
                        f"Step {global_step}: ↑performance, ↓guidance "
                        f"(ratio={ratio:.2f})"
                    )

        # 为下一周期准备，只留下当前步
        self._prev_accuracy_rewards = [current_step_accuracy] # 只保留当前步的准确率，如果下一步是3步周期中的第一步，会执行和这一句话一样的操作，等于多赋值了一次，不过不影响结果；
                                                              # 如果为3步周期中的第二步，那么下一步会将其步得到的accuracy_reward添加到列表尾部，这样一来就等于累积了3步周期中的第一步和第二步的accuracy_reward
        if self.accelerator.is_main_process and new_guidance_length == self.guidance_length:
            print(f"Step {global_step}: Performance stable, guidance unchanged")

        return new_guidance_length
    ###############################################################################################################################################################

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # 如果 guidance_length 是 None，则进行初始化，以防止 TypeError
        if self.guidance_length is None:
            self.guidance_length = 3072
        
        # 在步骤开始时锁定 guidance_length
        if self._step_guidance_length is None or self.state.global_step != getattr(self, '_last_step', -1):
            self._step_guidance_length = self.guidance_length
            self._last_step = self.state.global_step

            # 在新步骤开始时重置累积列表
            if not hasattr(self, '_step_accuracies'):
                self._step_accuracies = []
            else:
                self._step_accuracies.clear()

        current_guidance_length = self._step_guidance_length

        # print(">>> In GRPOTrainer._prepare_inputs, tokenizer padding_side =", self.processing_class.padding_side) # 发生更改
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # 发生更改，保存一个prompts_text的副本方便以后还原
        original_prompts_text = prompts_text.copy()

        # for i, prompt in enumerate(prompts_text[:5]):  # 打印前 5 个，确保涵盖 <|assistant|>
        #     print(f"DEBUG: prompts_text[{i}] = {repr(prompt)}") # 发生更改

        # 发生更改
        # 取出相应字段
        # deepseeks = [x["deepseek_attempt"] for x in inputs]
        # geminis = [x["gemini_attempt"] for x in inputs]
        # solution = [x["solution"] for x in inputs]
        generations = [x["generations"] for x in inputs]

        # 截取前面的tokens，因为tokens不像characters，所以需要转化一下再截取
        # 修改 truncate_text 函数，使其自动处理 token 长度判断
        def truncate_text(text, max_tokens=50):
            tokens = self.processing_class.encode(text, add_special_tokens=False)  # 先获取完整 token 列表
            if len(tokens) > max_tokens:  # 只有当 token 数超过 50 时才截取
                tokens = tokens[:max_tokens]
            elif len(tokens) < max_tokens:
                tokens += [self.processing_class.pad_token_id] * (max_tokens - len(tokens))  # 不足 50 的补 pad_token_id
            return tokens

        # If you still recall, "processing_class" will be assigned with a tokenizer if its value is "None" (See the 266^th line)
        # Generally, it will be pointed as a tokenizer when calling the GRPOTrainer class. See the file "grpo.py".
        # prompt_inputs = self.processing_class(
        #     prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        # )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)
        # prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)  # Gather the data across devices together.

            # # 验证分组情况（只打印前几个组）
            # G = self.num_generations
            # num_groups_to_show = 3

            # print("\n=== Prompt Groups (First few) ===")
            # for i in range(num_groups_to_show):
            #     group = all_prompts_text[i * G: (i + 1) * G]
            #     print(f"Group {i}:")
            #     for j, prompt in enumerate(group):
            #         print(f"  [{i * G + j}] {prompt}")

            generations = gather_object(generations)

            if self.accelerator.is_main_process:
                # 我认为拼接应该发生的地点
                # 用一个列表存储x的不同值
                completion_ids = []

                x_list = [current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          current_guidance_length, current_guidance_length, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                global_step = self.state.global_step

                print(f"Debug: [Step {global_step}] x_list = {x_list}")

                appended_ids_list = []
                appended_tokens_list = []
                prompt1_list = []

                for x1, prompt1, ds in zip(x_list, all_prompts_text, generations):
                    appended_ids = truncate_text(ds, max_tokens=x1)
                    appended_ids = torch.tensor(appended_ids, dtype=torch.long, device=device)
                    appended_tokens = self.processing_class.decode(appended_ids, skip_special_tokens=True)

                    # 把appended_ids放到同一个GPU上
                    appended_ids = appended_ids.to(device)

                    prompt1 = prompt1 + appended_tokens

                    # 保存结果到列表中
                    appended_ids_list.append(appended_ids)
                    appended_tokens_list.append(appended_tokens)
                    prompt1_list.append(prompt1)

                    # 由于我们单个单个处理，completion_ids只有一个维度，因此其长度就是token的数量
                    print(f"Debug: The length of appended_ids: {len(appended_ids)}")

                # 这里得到了整个的appended_ids_list和appended_tokens_list列表
                outputs = self.llm.generate(prompt1_list, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]

                for i in range(len(completion_ids)):
                    # 把 completion_ids[i] 变成 tensor
                    comp_tensor = torch.tensor(completion_ids[i], device=device)
                    # appended_ids 是形状 (batch_size, 50) 的张量，所以取 appended_ids[i]
                    appended_tensor = appended_ids_list[i]  # 形状 (50,) 注意这里添加的是有pad的ids，而不是除去了pad的tokens。

                    # 拼接
                    new_comp = torch.cat([appended_tensor, comp_tensor],
                                         dim=0)  # 比方说我prompt+引导的tokens有8k，在generate里面新生成了8k。我原本的appended_ids设置的是16k，那么实际上这里是24k的长度
                    # 为了方便后续 broadcast_object_list，需要把它转回 list
                    completion_ids[i] = new_comp.tolist()

            else:
                completion_ids = [None] * len(all_prompts_text)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # 发生更改，将prompt用于生成之后再转化为只有问题的prompt
            prompts_text = original_prompts_text

            # 发生更改
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids,
                                 padding_value=self.processing_class.pad_token_id)  # 把所有的completions补充到同一长度
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token (End Of Sequence)
        is_eos = completion_ids == self.processing_class.eos_token_id

        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 在这里新增一行，专门屏蔽 pad_token 的位置
        pad_positions = completion_ids.eq(self.processing_class.pad_token_id)  # 这里会专门找到padding token的位置
        completion_mask[pad_positions] = 0  # 创建一个mask矩阵，在padding token的位置上设置为0

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids,
                                                              skip_special_tokens=True)  # 但是到这一步，completion_ids中间的pad token会被跳过，原来24k长度的completion_ids会回到16k的一个长度
        # "is_conversational" is a format check.
        if is_conversational(inputs[0]):  # "[0]" means that you could retrieve any element in the variable.
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # In the following code snippet, the general process is: get "messages", use "message" to get "texts", use "texts" to compute rewards.
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs),
                                       device=device)  # 这个就是一个二维张量，旨在每一个completion都分配其所有的reward。比方说我对于一个prompt，其对应
        # 生成了5个completions，那么这个张量就是[5, 3]的形状，5是生成的completions的数量，3是reward函数的数量。
        # len(prompts)实际上是per_device_train_batch_size

        accuracy_reward_func = torch.zeros(len(prompts), device=device)  # 这个是用来存储accuracy的reward的

        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)  # reward_funcs是一个list，里面存放的是不同的reward函数
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False  # 发生更改
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:  # 由于我们只用简单函数作为reward，因此我们只看这一板块
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt",
                                                                "completion"]]  # 其实就是取出数据集中除开"prompt"和"completion"的字段名字，比方说"answer"和"solution"。
                reward_kwargs = {key: [example[key] for example in inputs] for key in
                                 keys}  # 收集数据集中的上述字段及其内容，打包成一个字典，例如：reward_kwargs = { "answer": [1, 2], "solution": ["A", "B"] }
                output_reward_func = reward_func(prompts=prompts, completions=completions,
                                                 **reward_kwargs)  # 可以参考reward.py中的各个reward的具体实现。记住reward_func是一个reward_funcs中的元素，也是一个函数
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32,
                                                      device=device)  # 这里对各个completion是批量处理的，反而是对于每种reward是循环单独处理的
                ####################################################################################我自己添加的用来求accuracy reward的片段#################################################################################
                if reward_func.__name__ == "accuracy_reward":
                    print(f"Step {self.state.global_step}: Raw accuracy_reward output = {output_reward_func}")
                    print(f"Step {self.state.global_step}: accuracy_reward tensor = {rewards_per_func[:, i]}")
                    # 这里的accuracy_reward是一个函数，返回的是一个list
                    accuracy_reward_func = torch.tensor(output_reward_func, dtype=torch.float32,
                                                        device=device)  # 这个就是一个一维张量，其长度是len(prompts)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)  # Cross different devices.
        # 那么此时rewards_per_func的尺寸就是[num_devices * per_device_train_batch_size, len(reward_funcs)]
        accuracy_reward_tensor = gather(accuracy_reward_func)  # 和整体的reward_per_func一样，这里也把所有设备上的accuracy_reward都收集到一起

        current_micro_batch_accuracy = accuracy_reward_tensor.mean().detach()

        if self.accelerator.is_main_process:
            print(f"\n=== Step {self.state.global_step} Accuracy Debug ===")
            print(f"accuracy_reward_tensor shape: {accuracy_reward_tensor.shape}")
            print(f"accuracy_reward_tensor values: {accuracy_reward_tensor}")
            print(f"accuracy_reward_tensor device: {accuracy_reward_tensor.device}")
            
            # 检查每个进程的贡献
            if accuracy_reward_tensor.dim() > 1:
                for proc_idx in range(accuracy_reward_tensor.shape[0]):
                    print(f"Process {proc_idx} accuracy: {accuracy_reward_tensor[proc_idx]}")
            
            # 计算总体统计
            print(f"Min accuracy: {accuracy_reward_tensor.min().item()}")
            print(f"Max accuracy: {accuracy_reward_tensor.max().item()}")
            print(f"Mean accuracy: {accuracy_reward_tensor.mean().item()}")
            print(f"Final current_step_accuracy: {current_micro_batch_accuracy.item()}")
            print("=" * 50)

        # 修正：将当前 micro-batch 的准确率添加到列表中，而不是覆盖
        if not hasattr(self, '_step_accuracies'): # 如果没发现这个属性，则创建这个属性，并且将其初始值赋为空列表
            self._step_accuracies = []
        self._step_accuracies.append(current_micro_batch_accuracy.item()) # 如果有这个属性，那么把当前micro step的accuracy_reward添加到这个列表中
        
        if self.accelerator.is_main_process:
            print(f"Step {self.state.global_step}: Added micro-batch accuracy {current_micro_batch_accuracy.item():.4f}")
            print(f"Step {self.state.global_step}: Current step accuracies so far: {self._step_accuracies}")

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(
            dim=1)  # unsqueeze只针对self.reward_weights，请注意
        """
        本来比方说：
        [[1., 2., 3.],
         [2., 3., 4.]]
        这个张量，unsqueeze之后就变成了：
        [
         [[1., 2., 3.],
          [2., 3., 4.]]
        ]
        那么此时在dim=1上求平均就变成了：[[1.5000, 2.5000, 3.5000]]，所以rewards的尺寸是从[num_devices * per_device_train_batch_size, len(reward_funcs)]进行了一个dim=1的求和，变成了[num_devices * per_device_train_batch_size]
        也就是说，rewards是每一个completion都求出它的各个reward之和（注意这里是sum(d=1)不是mean(dim=1)）。
        """

        """
        首先，每一步的reward都是要在当前步结束之后我们才会知道的。所以在这一步开始的时候，我们只能得知前面的accuracy_reward。因此，当前如果要根据accuracy_reward
        来选择当前步数的引导长度，肯定是不现实的。只能有两种方法，第一，根据上一步的accuracy_reward与上上、上上上步的accuracy_reward平均值来选择当前步数的引导长度；
        第二，我们在当前步的最开始拿到了上两步的平均accuracy_reward，在当前步的最后我们拿到了当前步的accuracy_reward。这样我们虽然当前步的引导选择已经来不及了，但是
        可以用来作为下一步的引导选择。很明显，第二个方法要简单且合理一些。由于当前步使用了上一步提供的引导选择，因此引导长度需要有一个初始值。
        """

        """
        real_step = self.state.global_step

        if real_step == 0:
            self._prev_accuracy_rewards = current_step_accuracy

        elif real_step % 2 != 0:
            self._prev_accuracy_rewards += current_step_accuracy

        elif real_step % 2 == 0:  # 每两步比较一次
            avg_prev_two_steps = self._prev_accuracy_rewards / 2.0

            if current_step_accuracy < avg_prev_two_steps:
                ratio = (avg_prev_two_steps / current_step_accuracy).item()
                self.guidance_length = int(min(self.guidance_length * ratio, self.max_guidance_length))
            elif current_step_accuracy > avg_prev_two_steps:
                ratio = (avg_prev_two_steps / current_step_accuracy).item()
                self.guidance_length = int(max(self.guidance_length * ratio, 1))
            else:
                self.guidance_length = self.guidance_length
            self._prev_accuracy_rewards = current_step_accuracy
        """

        #################################################################################################################################################################################################

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # 这个是用来求advantage求loss的。
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (
                std_grouped_rewards + 1e-4)  # Note that "rewards" here is a tensor, which contains all the rewards of all completions.
        # This is also the reason why we need to repeat_interleave the mean and std values before.
        # Slice to keep only the local part of the data
        # Cut the rewards and assign them to the corresponding devices, whose information will be given by ‘accelerator.process_index“。
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
                self.log_completions
                and self.state.global_step % self.args.logging_steps == 0
                and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1)  # we only need to compute the logits for the completion tokens 通俗点来说就是completion的token数，这个变量是为了在后面从整个prompt+completion中取出completion的一个标量
        # 如果这里是直接接收的_prepare_inputs返回的结果，那么completion_ids的长度还是24k
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask,
                                                    logits_to_keep)  # 注意这里由于completion_mask介入的缘故，padding token会被跳过

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        # print(">>> Using GRPOTrainer.prediction_step!") # 发生更改
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        ###################################################################################################################
        # 添加准确率对比日志
        if (self.accelerator.is_main_process and 
            "rewards/accuracy_reward" in metrics and 
            hasattr(self, '_step_accuracies') and self._step_accuracies):
            
            step_avg_accuracy = sum(self._step_accuracies) / len(self._step_accuracies)
            print(f"\n=== Accuracy Comparison (Step {self.state.global_step}) ===")
            print(f"Logged accuracy_reward: {metrics['rewards/accuracy_reward']:.4f}")
            print(f"Step average accuracy: {step_avg_accuracy:.4f}")
            print(f"Micro-batch accuracies: {self._step_accuracies}")
            print(f"Difference: {abs(metrics['rewards/accuracy_reward'] - step_avg_accuracy):.4f}")
            print("=" * 50)
        ###################################################################################################################

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))