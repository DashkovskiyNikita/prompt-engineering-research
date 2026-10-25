import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from datetime import datetime

API_KEY = ""
MODEL = "deepseek-flash" 
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

CONCURRENT_LIMIT = 50 

MAX_RETRIES = 5 
INPUT_FILE = "prompts_output/prompt_variations.json"
OUTPUT_FILE = "deepseek_results.json"
LOG_FILE = "deepseek_requests.log"

logger = logging.getLogger("deepseek_client")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


async def fetch_with_retry(session, prompt_data, semaphore, current_idx, total_tasks):
    task_id = prompt_data["task_id"]
    method = prompt_data["method"]
    prompt = prompt_data["prompt"]
    log_prefix = f"[{current_idx}/{total_tasks} | {task_id}]"
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    BASE_URL,
                    headers=headers,
                    json=payload
                ) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2))
                        sleep_time = min(retry_after + 1, 5)
                        logger.warning(f"{log_prefix} Rate limit (429). Ожидание {sleep_time}с. (Попытка {attempt + 1}/{MAX_RETRIES})")
                        await asyncio.sleep(sleep_time)
                        continue
                    
                    resp.raise_for_status()
                    result = await resp.json()
                    
                    message = result["choices"][0]["message"]
                    content = message.get("content", "")
                    reasoning = message.get("reasoning_content", "")
                    
                    logger.debug(f"{log_prefix} Успех с {attempt + 1}-й попытки.")
                    
                    return {
                        "task_id": task_id,
                        "method": method,
                        "prompt": prompt,
                        "response": content,
                        "reasoning": reasoning,
                        "status": "success",
                        "attempts": attempt + 1
                    }
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"{log_prefix} Критическая ошибка после {MAX_RETRIES} попыток: {e}")
                    return {
                        "task_id": task_id,
                        "method": method,
                        "prompt": prompt,
                        "error": str(e),
                        "status": "failed",
                        "attempts": attempt + 1
                    }
                
                sleep_time = 2
                logger.warning(f"{log_prefix} Ошибка: {e}. Повтор через {sleep_time}с. (Попытка {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(sleep_time)


async def main():
    logger.info("=" * 60)
    logger.info("Запуск скрипта оценки промптов через DeepSeek API")
    logger.info(f"Модель: {MODEL}, Лимит одновременных запросов: {CONCURRENT_LIMIT}")
    logger.info("=" * 60)

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        logger.error(f"Файл не найден: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    
    total_tasks = len(prompts_data)
    logger.info(f"Загружено задач для обработки: {total_tasks}")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_LIMIT)
    
    start_time = datetime.now()
    
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_with_retry(session, item, semaphore, idx + 1, total_tasks)
            for idx, item in enumerate(prompts_data)
        ]
        
        results = await asyncio.gather(*tasks)

    end_time = datetime.now()
    duration = end_time - start_time

    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    success_count = sum(1 for r in results if r["status"] != "failed")
    failed_count = total_tasks - success_count
    
    logger.info("=" * 60)
    logger.info("Обработка завершена!")
    logger.info(f"Всего задач: {total_tasks}")
    logger.info(f"Успешно: {success_count}")
    logger.info(f"Ошибки: {failed_count}")
    logger.info(f"Затраченное время: {duration}")
    logger.info(f"Результаты сохранены в: {output_path}")
    logger.info(f"Полный лог сохранен в: {LOG_FILE}")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())