import json
import time
from multiprocessing import Pool, Manager
from functools import partial
from openai import OpenAI
from tqdm import tqdm

def make_api_call(args_tuple, result_queue):
    """Make a single API call and put result in queue"""
    messages, index = args_tuple
    client = OpenAI(api_key='', base_url='http://127.0.0.1:8000/v1')
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=messages,
        n=1,
        timeout=600
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

def main():
    all_messages = []
    with open("data/code_1.jsonl", 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            data = json.loads(line)
            messages = [{"role": "user", "content": data['instruction']}]
            all_messages.append(messages)
    
    with Manager() as manager:
        result_queue = manager.Queue()
        
        process_func = partial(make_api_call, result_queue=result_queue)
        
        with Pool(processes=32) as pool:
            task_count = len(all_messages)
            results = pool.imap_unordered(
                process_func,
                [(msg, idx) for idx, msg in enumerate(all_messages)]
            )
            
            all_messages_out = []
            for _ in tqdm(results, total=task_count, desc="API calls"):
                all_messages_out.append(result_queue.get())
        
        all_messages_out.sort(key=lambda x: x['index'])
        
        for item in all_messages_out:
            del item['index']
        
        with open("data/code_1_out.json", 'w', encoding='utf-8') as f:
            json.dump(all_messages_out, f, ensure_ascii=False, indent=4)
        
        print(f"Processed {len(all_messages_out)} messages successfully")

if __name__ == '__main__':
    main()