from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
