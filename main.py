# Chạy trên CPU hoàn toàn
import librosa
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeProcessor

# 1. Khởi tạo (chỉ cần CPU)
processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")

# 2. Xử lý template văn bản
audio_path = "test.mp3" 

messages = [
    {
        "role": "system",
        "content": """You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{
    'type': 'function', 
    'function': {
        'name': 'escalate_claim_discrepancy', 
        'description': 'Escalate an insurance claim to a manager when there is a payout discrepancy or customer dissatisfaction.', 
        'parameters': {
            'type': 'object', 
            'properties': {
                'claim_id': {
                    'type': 'string', 
                    'description': 'The unique identifier for the insurance claim (e.g., AUTO-998877)'
                },
                'issue_type': {
                    'type': 'string', 
                    'enum': ['payout_discrepancy', 'repair_vs_replacement', 'denied_claim'],
                    'description': 'The primary nature of the claim issue'
                },
                'customer_sentiment': {
                    'type': 'string', 
                    'description': 'The emotional tone of the customer (e.g., frustrated, angry, neutral)'
                },
                'request_manager': {
                    'type': 'boolean', 
                    'description': 'Whether the customer explicitly asked to speak with or escalate to a manager'
                }
            }, 
            'required': ['claim_id', 'issue_type', 'request_manager']
        }
    }
}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path}
        ]
    }
]

text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f"Text Prompt: {text_prompt}")

# 3. Xử lý audio thô sang Numpy array
# (Hàm này gọi librosa.load bên trong)
audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
print(f"Audio array shape: {audios[0].shape}") # Bạn sẽ thấy mảng số thực ở đây

# 4. Bước cuối: Tạo input cho LLM (vẫn chạy được trên CPU)
inputs = processor(text=text_prompt, audio=audios, return_tensors="pt")
print(f"Keys gửi vào LLM: {inputs.keys()}")