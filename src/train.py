import unsloth
from unsloth import FastLanguageModel
import torch
import os
from trl import GRPOConfig, GRPOTrainer
import click
import wandb
import numpy as np
import json
import subprocess
from datasets import Dataset
import random
import string
from transformers import AutoTokenizer
import re

def generate_mapping(num_inputs: int, num_outputs: int, seed: int = 0) -> dict[str, int]:
    """
    Generate a seeded random mapping from input letters to output numbers.
    
    Args:
        num_inputs (int): Number of input letters (A..)
        num_outputs (int): Number of output numbers (0..num_outputs-1)
        seed (int): Random seed for reproducibility

    Returns:
        dict: Mapping from input letter -> output number
    """
    random.seed(seed)

    letters = list(string.ascii_uppercase[:num_inputs])
    numbers = list(range(num_outputs))

    mapping = {letter: random.choice(numbers) for letter in letters}
    return mapping


def generate_dataset(mapping: dict[str, int], num_outputs: int, ask_all: bool = False) -> Dataset:
    """
    Given a letter->number mapping, generate a HuggingFace Dataset
    of (prompt, response) pairs suitable for GRPO fine-tuning.
    """
    letters = sorted(mapping.keys())
    # Use all possible numbers, not just those appearing in the mapping
    numbers = list(range(num_outputs))

    if ask_all:
        instr = 'Determine what all the leters map to. Answer in 50 words or less. Put your final answer within \\boxed{{}}. Give your answer as \\boxed{{A=<NUM>,B=<NUM>, etc}}. Do this in the order that the letters are given.'
    else:
        instr = 'Determine what the below letter maps to. Answer in 50 words or less. Put your final answer within \\boxed{{}}.'

    base_prompt = (
        "You have the below input letters that each can map to one of the output numbers. "
        "Multiple letters can map to the same number, and vice versa.\n"
        f"Letters: {{{', '.join(letters)}}}\n"
        f"Numbers: {{{', '.join(map(str, numbers))}}}\n"
        f"{instr}\n"
    )
    samples = []
    if ask_all:
        letter_prompt = ''
        answer = []
        for letter in letters:
            letter_prompt += f"{letter}=\n"
            answer.append(mapping[letter])
        prompt = base_prompt + letter_prompt[:-1]
        samples.append({"question": prompt, "answer": answer})
    else:
        for letter in letters:
            prompt = base_prompt + f"{letter}="
            answer = mapping[letter]
            samples.append({"question": prompt, "answer": answer})

    return Dataset.from_list(samples)


def extract_boxed_content(text: str) -> int:
    """
    Extracts the last value found inside LaTeX-style \\boxed{...} blocks.

    Args:
        text (str): The full text from the LLM output.

    Returns:
        Optional[int]: The last boxed value, or None if none found.
    """
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    try:
        return int(str(matches[-1]).strip().lower())
    except:
        return None

def extract_all_mapping_answer(text: str) -> list[int]:
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    try:
        s = str(matches[-1]).strip()
        mapping = []
        for pair in s.split(","):
            try:
                if "=" in pair:
                    key, value = pair.split("=")
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    mapping.append(value)
            except:
                mapping.append(None)
        return mapping
    except:
        return None

def create_reward_fn(regression_reward, ask_all, num_inputs):
    def _reward_fn(completions, answer, **kwargs):
        scores = []
        for ans, comp in zip(answer, completions):
            pred = extract_boxed_content(comp)
            ans = int(ans)
            if regression_reward:
                if pred is None:
                    scores.append(-100000)
                else:
                    scores.append(
                        -np.abs(pred-ans)
                    )
            else:
                scores.append(pred == ans)
        return np.array(scores).astype(int)
    
    def _ask_all_reward_fn(completions, answer, **kwargs):
        scores = []
        for ans, comp in zip(answer, completions):
            pred_list = extract_all_mapping_answer(comp)
            if pred_list is None:
                scores.append(-1)
            else:
                assert len(pred_list) == len(ans), (len(pred_list), len(ans))
                score = 0
                for p, a in zip(pred_list, ans):
                    if p == a:
                        score += 1 / num_inputs
            scores.append(score)
        return np.array(scores).astype(int)
    if ask_all:
        return _ask_all_reward_fn
    else:
        return _reward_fn

# ---------- Main Functions ----------
def load_train_model_and_tokenizer(model_name, max_seq_length: int = 2048, lora_rank: int = 32, load_in_4bit = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        #fast_inference=True,
        #max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer

def train(model,
          tokenizer,
          dataset,
          run_name: str,
          reward_fn, 
          max_completion_length: int = 250,
          num_generations: int = 8,
          batch_size: int = 4,
          max_steps: int = 1000,
          checkpoint_dir: str = 'runs',
          save_steps: int = 100,
          beta: float = 0.001):
    config = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=checkpoint_dir,
        run_name=run_name,
        save_steps=save_steps,
        beta=beta
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_fn,
        ],
        args=config,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    model.save_pretrained(f'{checkpoint_dir}/final')

def setup_wandb(project, name, skip_train):
    if skip_train:
        # resume previous run
        api = wandb.Api()
        runs = api.runs(f"{wandb.api.default_entity}/{project}")
        matched = [run.id for run in runs if run.name == name]
        if len(matched) == 0:
            raise ValueError(f"No W&B run with name '{name}' found in project '{project}'")
        id_ = matched[-1]
        print(f'Resume run {name} with id {id_}')
        wandb.init(project=project, name=name, id = id_, resume="must")
    else:
        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_NAME'] = name

        # calling init now to save both train and test
        wandb.init(
            project=project,
            name=name
        )

# def log_inference_results(results_path):
#     """Log inference results to the active wandb run"""
#     if wandb.run is None:
#         print("Warning: No active wandb run found for logging inference results")
#         return
    
#     if not os.path.exists(results_path):
#         raise ValueError(f'{results_path} not found')
    
#     with open(results_path) as f:
#         results = json.load(f)
    
#     # Extract data from results dictionary
#     checkpoint_numbers = results.get('checkpoint', [])
#     accuracies = results.get('accuracy', [])
#     pass_at_k_key = [k for k in results.keys() if k.startswith('pass@')][0] if any(k.startswith('pass@') for k in results.keys()) else None
#     pass_at_k_values = results.get(pass_at_k_key, []) if pass_at_k_key else []
    
#     final_acc = accuracies[-1]
#     best_acc = max(accuracies)
#     final_pass = pass_at_k_values[-1]
#     best_pass = max(pass_at_k_values)

#     base_accuracy = results.get('base accuracy', 0)
#     base_pass_at_k_key = [k for k in results.keys() if k.startswith('base pass@')][0] if any(k.startswith('base pass@') for k in results.keys()) else None
#     base_pass_at_k = results.get(base_pass_at_k_key, 0) if base_pass_at_k_key else 0
    
#     metric_dict =  {
#         "final_accuracy": final_acc,
#         "best_accuracy": best_acc,
#         "base_acc": base_accuracy,
#         f'final {pass_at_k_key}' : final_pass,
#         f'best {pass_at_k_key}' : best_pass,
#         f'base {pass_at_k_key}' : base_pass_at_k, 
#         'checkpoints': checkpoint_numbers,
#         'accuracies' : accuracies,
#         pass_at_k_key: pass_at_k_values
#     }

#     wandb.run.summary.update(metric_dict)

#     print(f"Logged inference results for {len(checkpoint_numbers)} checkpoints to wandb")

def _get_base_path():
    return os.path.dirname(os.path.abspath(__file__))

def _get_checkpoint_dir(name: str):
    checkpoint_base_dir = os.path.join(_get_base_path(), 'checkpoints')
    return os.path.join(
        checkpoint_base_dir,
        name
    )

def format_single_question(question: str, tokenizer: AutoTokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 
          'content': question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def format_dataset_(dataset, tokenizer: AutoTokenizer):
    def _format_prompt(example):
        new_prompt = format_single_question(
            example['question'],
            tokenizer,
        )
        return {'prompt': new_prompt}
    dataset = dataset.map(_format_prompt)
    dataset = dataset.remove_columns(['question'])
    return dataset

@click.command()
@click.option(
    '--num_input',
    '-i',
    type=int,
    required=True,
    help='Number of input letters'
)
@click.option(
    '--num_output',
    '-o',
    type=int,
    required=True,
    help='Number of output numbers'
)
@click.option('--project', type=str, default='GRPO_bits')
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-8B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option('--max_steps',
              type=int,
              default=500,
              help='Number of generations per iteration')
@click.option('--save_steps',
              type=int,
              default=100,
              help='How often to save')
@click.option('--seed',
              type=int,
              default=42,
              help='Seed to use')
@click.option('--regression_reward', '-r', is_flag=True, default=False,
    help="Modify the reward to be regression rather than 0/1")
@click.option('--ask_all', is_flag=True, default=True,
    help="Ask for all at the same time")
def main(
    num_input: int,
    num_output: int,
    project: str,
    num_generations: int,
    model_name: str,
    max_steps: int,
    save_steps: int,
    seed: int,
    regression_reward: bool,
    ask_all: bool
):
    name = f'input{num_input}_output{num_output}_{model_name}_seed{seed}_{num_generations}gen_{max_steps}steps_nomath'
    if regression_reward:
        click.echo(f"Using regression reward")
        name += '_regressionreward'
    if ask_all:
        click.echo(f'Using ask all')
        name += '_askall'
    setup_wandb(project=project, name=name, skip_train=False)

    mapping = generate_mapping(
        num_inputs=num_input,
        num_outputs=num_output,
        seed=seed
    )
    click.echo(f'Mapping: {mapping}')

    dataset = generate_dataset(mapping, num_output, ask_all=ask_all)
    click.echo(f'Loaded train dataset of size {len(dataset)}')

    checkpoint_dir = _get_checkpoint_dir(name)
    click.echo(f'Checkpoint directory: {checkpoint_dir}')

    model, tokenizer = load_train_model_and_tokenizer(model_name=model_name)
    dataset = format_dataset_(dataset, tokenizer)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(
        model=model,
        tokenizer=tokenizer, 
        dataset=dataset,
        run_name=name,
        reward_fn=create_reward_fn(regression_reward, ask_all=ask_all, num_inputs=len(mapping)),
        num_generations=int(num_generations),
        batch_size=1,
        max_steps=max_steps,
        save_steps=save_steps,
        checkpoint_dir=checkpoint_dir,
        #beta=beta
    )
    
    # clear up memory before inference
    # model.to('cpu')
    # del model
    # del tokenizer
    # torch.cuda.empty_cache()

    # # Run inference - determine correct path to get_answers.py
    # if os.path.exists('get_answers.py'):
    #     get_answers_path = 'get_answers.py'
    # elif os.path.exists('src/get_answers.py'):
    #     get_answers_path = 'src/get_answers.py'
    # else:
    #     click.echo('Error: Cannot find get_answers.py')
    #     return
    
    # cmd = f'python {get_answers_path} -m {model_name} --split test --dataset_name {dataset_name} -b {test_batch_size} --num_repeat {test_num_repeat} --run_name {name}'
    # if eval_last:
    #     cmd += ' --eval_last'
    # click.echo(f'Running command: {cmd}')
    # subprocess.run(cmd, shell=True)

    # # Log inference results to the same wandb run
    # log_inference_results(
    #     os.path.join(checkpoint_dir, 'test_results.json')
    # )

if __name__ == '__main__':
    main()