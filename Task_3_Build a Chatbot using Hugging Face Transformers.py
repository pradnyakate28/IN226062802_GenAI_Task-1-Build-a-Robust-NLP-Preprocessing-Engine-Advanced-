# Step 1: Install libraries
!pip install transformers
!pip install torch

===========================================================================================================================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 2: Load DialoGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Step 3: Simple factual knowledge base
fact_knowledge = {
    "who created python": "Python was created by Guido van Rossum and released in 1991.",
    "what is python": "Python is a high-level, interpreted programming language created by Guido van Rossum.",
    "what is artificial intelligence": "Artificial Intelligence refers to the simulation of human intelligence by machines that can perform tasks such as learning, reasoning, and problem solving.",
    "who is guido van rossum": "Guido van Rossum is the creator of Python programming language.",
    "what is machine learning": "Machine Learning is a subset of AI that enables systems to learn from data and improve from experience without being explicitly programmed."
}

# Step 4: Initialize chat history
chat_history_ids = None

# Step 5: Start chatbot
print("Chatbot: Hello! I am your AI assistant. Type 'exit' or 'quit' to stop.")

while True:
    user_input = input("User: ").strip()

    # Step 5.1: Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! Have a nice day.")
        break

    # Step 5.2: Check factual knowledge first
    lower_input = user_input.lower()
    response = None
    for key in fact_knowledge:
        if key in lower_input:
            response = fact_knowledge[key]
            break

    # Step 5.3: If not a factual question, use DialoGPT
    if response is None:
        prompt_text = user_input  # keep normal prompt
        new_input_ids = tokenizer.encode(prompt_text + tokenizer.eos_token, return_tensors='pt')

        # Append previous chat history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate model response
        chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=torch.ones_like(bot_input_ids),  # fix attention warning
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )

        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Step 5.4: Print response
    print(f"Chatbot: {response}")

===========================================================================================================================================================================

Loading weights:   0%|          | 0/293 [00:00<?, ?it/s]
The tied weights mapping and config for this model specifies to tie transformer.wte.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
GPT2LMHeadModel LOAD REPORT from: microsoft/DialoGPT-medium
Key                              | Status     |  | 
---------------------------------+------------+--+-
transformer.h.{0...23}.attn.bias | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Chatbot: Hello! I am your AI assistant. Type 'exit' or 'quit' to stop.
User: Who created python?
Chatbot: Python was created by Guido van Rossum and released in 1991.
User: What is Artificial Intelligence?
Chatbot: Artificial Intelligence refers to the simulation of human intelligence by machines that can perform tasks such as learning, reasoning, and problem solving.
User: What is machine learning?
Chatbot: Machine Learning is a subset of AI that enables systems to learn from data and improve from experience without being explicitly programmed.
User: How are you?
Chatbot: Fine. You?
User: exit
Chatbot: Goodbye! Have a nice day.