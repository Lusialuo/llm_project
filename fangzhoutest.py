import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# 使用 tenacity 库进行自动重试，解决 429 RateLimitExceeded 错误
# 策略：指数退避，最多重试 5 次，等待时间在 2 到 10 秒之间
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def call_model_with_retry(client, model, messages):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages
        )
    except Exception as e:
        print(f"Request failed: {e}, retrying...")
        raise e

# 优先使用环境变量，如果没有则使用硬编码的 key (来自 test.py)
api_key = os.getenv('ARK_API_KEY') or "598d7c22-d2af-4a92-bce2-5c9be3170b6a"

client = OpenAI(
    base_url="https://ark-cn-beijing.bytedance.net/api/v3",
    api_key=api_key,
)

# 修正为标准的 OpenAI Chat 格式
response = call_model_with_retry(
    client,
    model="ep-20260212190221-rdt4x",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"
                    }
                },
                {
                    "type": "text",
                    "text": "你看见了什么？"
                },
            ],
        }
    ]
)

print(response.choices[0].message.content)
