import os
import boto3

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "deepseek.v3.2")
REGION = os.getenv("BEDROCK_REGION", "us-east-2")

print(f"Testing model: {MODEL_ID} in {REGION}")

client = boto3.client('bedrock-runtime', region_name=REGION)

response = client.converse(
    modelId=MODEL_ID,
    messages=[
        {"role": "user", "content": [{"text": "Hola, ¿cuánto es 2+2?"}]}
    ],
    inferenceConfig={"temperature": 0, "maxTokens": 100},
)

print(response['output']['message']['content'][0]['text'])
