from transformers import AutoTokenizer, MllamaForCausalLM, AutoModel
model = MllamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
prompt = "If I had to write a haiku, it would be:"
inputs = tokenizer(prompt, return_tensors="pt")  # dict_keys(['input_ids', 'attention_mask'])


outputs = model(inputs.input_ids, inputs.attention_mask, output_hidden_states=True) # odict_keys(['last_hidden_state', 'past_key_values', 'hidden_states'])

from transformers import PaliGemmaForConditionalGeneration
model = PaliGemmaForConditionalGeneration.from_pretrained('google/paligemma2-3b-pt-224')
tokenizer = AutoTokenizer.from_pretrained('google/paligemma2-3b-pt-224')

output = model.language_model(inputs.input_ids, inputs.attention_mask, output_hidden_states=True)
