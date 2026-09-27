from transformers import AutoTokenizer
import csv

# ==========================================
# 1. LOAD BERT TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("========== BERT TOKENIZER ==========")
print("BERT Vocabulary Size:", tokenizer.vocab_size)


# ==========================================
# 2. SAMPLE TEXT
# ==========================================

text = "Hello, I am learning LLM tokenization."

print("\n========== INPUT TEXT ==========")
print(text)


# ==========================================
# 3. TOKENIZATION
# ==========================================

tokens = tokenizer.tokenize(text)

print("\n========== TOKENS ==========")
print(tokens)


# ==========================================
# 4. TOKEN IDs
# ==========================================

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n========== TOKEN IDs ==========")
print(token_ids)


# ==========================================
# 5. ENCODE WITH SPECIAL TOKENS
# ==========================================

encoded = tokenizer.encode(text)

print("\n========== ENCODED IDS ==========")
print(encoded)


# ==========================================
# 6. TOKENS WITH SPECIAL TOKENS
# ==========================================

encoded_tokens = tokenizer.convert_ids_to_tokens(encoded)

print("\n========== TOKENS WITH SPECIAL TOKENS ==========")
print(encoded_tokens)


# ==========================================
# 7. ATTENTION MASK
# ==========================================

encoded_data = tokenizer(
    text,
    padding="max_length",
    max_length=20,
    truncation=True
)

print("\n========== ATTENTION MASK ==========")
print(encoded_data["attention_mask"])


# ==========================================
# 8. SAVE TOKENIZED OUTPUT
# ==========================================

with open("tokenized_output.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(["Token", "Token ID"])

    for token, token_id in zip(tokens, token_ids):
        writer.writerow([token, token_id])

print("\n========== FILE SAVED ==========")
print("tokenized_output.csv created successfully!")