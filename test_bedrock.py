import boto3

client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-2',
)

MODEL_ID = "deepseek.v3.2"

response = client.converse(
    modelId=MODEL_ID,
    messages=[
        {"role": "user", "content": [{"text": "Hola, ¿cuánto es 2+2?"}]}
    ],
    inferenceConfig={"temperature": 0, "maxTokens": 100},
)

print(response['output']['message']['content'][0]['text'])
