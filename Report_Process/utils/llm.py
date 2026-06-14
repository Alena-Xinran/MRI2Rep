from litellm import completion
import config

def call_llm(prompt, temperature=0.0):
    """
    统一的 LLM 调用接口
    """
    try:
        response = completion(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant specialized in MRI report analysis. Follow instructions strictly."},
                {"role": "user", "content": prompt}
            ],
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            temperature=temperature,
            drop_params=True
        )
        return response
    except Exception as e:
        print(f"❌ LLM Call Error: {e}")
        return None
