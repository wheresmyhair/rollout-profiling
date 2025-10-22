import json
import time
import os
from multiprocessing import Pool, Manager
from functools import partial
from openai import OpenAI
from tqdm import tqdm

CHECKPOINT_DIR = "checkpoints"
SAVE_INTERVAL = 50  # Save every 50 items

def ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(dataset_name):
    """Get checkpoint file path for a dataset"""
    return os.path.join(CHECKPOINT_DIR, f"{dataset_name}_checkpoint.json")

def get_saved_indices(dataset_name):
    """Get set of indices that have been saved in checkpoint"""
    checkpoint_path = get_checkpoint_path(dataset_name)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                return set(item['index'] for item in results)
        except (json.JSONDecodeError, KeyError):
            return set()
    return set()

def load_checkpoint(dataset_name):
    """Load partial results from checkpoint"""
    checkpoint_path = get_checkpoint_path(dataset_name)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Checkpoint corrupted, starting fresh")
            return []
    return []

def save_checkpoint(dataset_name, results):
    """Save partial results to checkpoint"""
    checkpoint_path = get_checkpoint_path(dataset_name)
    # Write to temp file first, then rename (atomic write)
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Atomic rename
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    os.rename(temp_path, checkpoint_path)

def make_api_call(args_tuple, result_queue):
    """Make a single API call and put result in queue"""
    messages, index = args_tuple
    client = OpenAI(api_key='', base_url='http://127.0.0.1:8000/v1')
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        n=1,
        timeout=9999
    )
    end_time = time.time()
    
    out_dict = {
        "index": index,
        "input": messages,
        "output": response.choices[0].message.content,
        "time": end_time - start_time,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    
    result_queue.put(out_dict)

def process_dataset(dataset_config):
    """Process a single dataset with checkpointing"""
    dataset_name = dataset_config["name"]
    input_file = dataset_config["input_file"]
    input_field = dataset_config["input_field"]
    output_file = dataset_config["output_file"]
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    ensure_checkpoint_dir()
    
    # Load checkpoint with already saved results
    all_messages_out = load_checkpoint(dataset_name)
    saved_indices = get_saved_indices(dataset_name)
    print(f"Loaded from checkpoint: {len(all_messages_out)} items")
    
    # Load data and filter out already processed
    all_messages = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, desc=f"Loading {dataset_name}")):
            if idx not in saved_indices:
                data = json.loads(line)
                messages = [{"role": "user", "content": data[input_field]}]
                all_messages.append((messages, idx))
    
    if not all_messages:
        print(f"✓ {dataset_name} already fully processed!")
        return
    
    print(f"Remaining to process: {len(all_messages)} items")
    
    with Manager() as manager:
        result_queue = manager.Queue()
        process_func = partial(make_api_call, result_queue=result_queue)
        
        with Pool(processes=64) as pool:
            task_count = len(all_messages)
            results = pool.imap_unordered(
                process_func,
                all_messages
            )
            
            items_since_save = 0
            for _ in tqdm(results, total=task_count, desc=f"API calls ({dataset_name})"):
                result = result_queue.get()
                all_messages_out.append(result)
                
                items_since_save += 1
                if items_since_save >= SAVE_INTERVAL:
                    save_checkpoint(dataset_name, all_messages_out)
                    print(f"  → Checkpoint saved ({len(all_messages_out)} total items)")
                    items_since_save = 0
    
    # Final sort and cleanup
    all_messages_out.sort(key=lambda x: x['index'])
    for item in all_messages_out:
        del item['index']
    
    # Save final output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_messages_out, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Saved {len(all_messages_out)} items to {output_file}")
    
    # Clean up checkpoint
    checkpoint_path = get_checkpoint_path(dataset_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"✓ Cleaned up checkpoint files")

def main():
    # Define all datasets
    datasets = [
        {
            "name": "code_1",
            "input_file": "data/code_1.jsonl",
            "input_field": "instruction",
            "output_file": "data/code_1_out.json",
        },
        {
            "name": "math_1",
            "input_file": "data/math_1.jsonl",
            "input_field": "question",
            "output_file": "data/math_1_out.json",
        },
        {
            "name": "math_2",
            "input_file": "data/math_2.jsonl",
            "input_field": "problem",
            "output_file": "data/math_2_out.json",
        },
        {
            "name": "math_3",
            "input_file": "data/math_3.jsonl",
            "input_field": "question",
            "output_file": "data/math_3_out.json",
        },
    ]
    
    total_processed = 0
    for dataset_config in datasets:
        process_dataset(dataset_config)
        total_processed += 1
    
    print(f"\n{'='*60}")
    print(f"✓ All {total_processed} datasets processed!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()