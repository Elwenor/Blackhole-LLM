# File: inference.py

import torch
import re
import numpy as np

# CORRECTED: Importy z nova_test_file
# Assuming decode_number_from_features will be available in modeling_nova.py or a dedicated embedding.py in nova_test_file
from nova_test_file.modeling_nova import decode_number_from_features # Adjust if decode_number_from_features is elsewhere
from nova_test_file.hugging_tokenizer2 import CUSTOM_SPECIAL_TOKENS
from nova_test_file.configuration_nova import BlackholeConfig

def predict_and_decode_answer(model, encoder_token_ids, encoder_numeric_features, encoder_attention_mask, tokenizer, device, max_decoding_len=128):
    """
    Generates an answer from the model given encoded input.
    Decodes predicted tokens and numerical features into a human-readable string.

    Args:
        model: The trained BlackholeSeq2SeqForConditionalGeneration model.
        encoder_token_ids: Token IDs for the encoder input.
        encoder_numeric_features: Numeric features for the encoder input.
        encoder_attention_mask: Attention mask for the encoder input.
        tokenizer: The BlackholeTokenizer used for encoding/decoding.
        device: The device (cuda/cpu) to run inference on.
        max_decoding_len: Maximum length of the generated sequence.
    """
    model.eval()

    num_token_id = tokenizer.convert_tokens_to_ids(CUSTOM_SPECIAL_TOKENS["number_token"])
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    unk_token_id = tokenizer.unk_token_id
    pad_token_id = tokenizer.pad_token_id

    numeric_feature_dim = model.config.numeric_input_features
    padded_feat_row = torch.full((numeric_feature_dim,), -2.0, dtype=torch.float32, device=device)

    batch_size = encoder_token_ids.size(0)
    if batch_size == 0:
        return [], []

    if torch.all(~encoder_attention_mask):
        placeholder = "<INPUT_FULLY_MASKED>"
        return [placeholder] * batch_size, [None] * batch_size

    decoder_input_token_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    decoder_input_numeric_features = padded_feat_row.unsqueeze(0).repeat(batch_size, 1, 1)

    generated_token_sequences = [[] for _ in range(batch_size)]
    generated_feature_sequences = [[] for _ in range(batch_size)]

    done_flags = [False] * batch_size

    with torch.no_grad():
        encoder_padding_mask = (encoder_attention_mask == 0)
        # Assuming _fuse_embeddings is a method of the model, not a standalone function
        encoder_input_embeds = model._fuse_embeddings(encoder_token_ids, encoder_numeric_features) 
        encoder_hidden_states = model.encoder(src=encoder_input_embeds, src_key_padding_mask=encoder_padding_mask)


        for _ in range(max_decoding_len):
            decoder_attention_mask = torch.ones_like(decoder_input_token_ids, dtype=torch.bool, device=device)
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_input_token_ids.size(1)).to(device)

            decoder_input_embeds = model._fuse_embeddings(decoder_input_token_ids, decoder_input_numeric_features)
            decoder_padding_mask = (decoder_attention_mask == 0)

            decoder_output = model.decoder(
                tgt=decoder_input_embeds,
                memory=encoder_hidden_states,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=encoder_padding_mask
            )

            token_logits = model.lm_head(decoder_output)
            num_feature_output = model.numeric_head(decoder_output)

            next_token_logits = token_logits[:, -1, :]
            next_num_features = num_feature_output[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1)

            decoder_input_token_ids = torch.cat([decoder_input_token_ids, next_token_ids.unsqueeze(1)], dim=1)

            next_step_features = torch.stack([
                next_num_features[b] if next_token_ids[b].item() == num_token_id else padded_feat_row
                for b in range(batch_size)
            ]).unsqueeze(1)
            decoder_input_numeric_features = torch.cat([decoder_input_numeric_features, next_step_features], dim=1)

            for i in range(batch_size):
                if not done_flags[i]:
                    token_id = next_token_ids[i].item()
                    if token_id == eos_token_id:
                        done_flags[i] = True
                    else:
                        generated_token_sequences[i].append(token_id)
                        generated_feature_sequences[i].append(next_num_features[i].cpu().numpy())

            if all(done_flags):
                break

    final_decoded_answers = []
    final_predicted_numbers = []

    cap_token = CUSTOM_SPECIAL_TOKENS["capitalized_token"] # Corrected key
    allcaps_token = CUSTOM_SPECIAL_TOKENS["all_caps_token"]
    space_token = CUSTOM_SPECIAL_TOKENS["space_token"]

    for i in range(batch_size):
        tokens = [tokenizer.convert_ids_to_tokens(tok_id) for tok_id in generated_token_sequences[i]]
        features = generated_feature_sequences[i]

        decoded_numbers = [decode_number_from_features(feat) for feat in features]
        last_valid_number = next((num for num in reversed(decoded_numbers) if not np.isnan(num)), None)
        final_predicted_numbers.append(last_valid_number)

        predicted_answer_tokens = []
        feature_idx = 0
        tok_idx = 0

        while tok_idx < len(tokens):
            token = tokens[tok_idx]

            if token == CUSTOM_SPECIAL_TOKENS["number_token"]:
                if feature_idx < len(decoded_numbers):
                    val = decoded_numbers[feature_idx]
                    if not np.isnan(val):
                        if abs(val - round(val)) < 1e-6:
                            predicted_answer_tokens.append(str(int(round(val))))
                        else:
                            predicted_answer_tokens.append(f"{val}")
                    else:
                        predicted_answer_tokens.append("<NUM>")
                else:
                    predicted_answer_tokens.append("<NUM>")
                feature_idx += 1

            elif token == cap_token and tok_idx + 1 < len(tokens):
                next_text_token = tokenizer.convert_ids_to_tokens(generated_token_sequences[i][tok_idx+1])
                predicted_answer_tokens.append(next_text_token.capitalize())
                tok_idx += 1
                feature_idx += 1

            elif token == allcaps_token and tok_idx + 1 < len(tokens):
                next_text_token = tokenizer.convert_ids_to_tokens(generated_token_sequences[i][tok_idx+1])
                predicted_answer_tokens.append(next_text_token.upper())
                tok_idx += 1
                feature_idx += 1

            elif token == space_token:
                predicted_answer_tokens.append(' ')
                feature_idx += 1

            elif tokenizer.convert_tokens_to_ids(token) in [pad_token_id, bos_token_id, eos_token_id, unk_token_id]:
                pass
            else:
                predicted_answer_tokens.append(token)
                feature_idx += 1

            tok_idx += 1

        predicted_answer_raw = "".join(predicted_answer_tokens)
        predicted_answer_cleaned = re.sub(r'\s+([.,!?;:])', r'\1', predicted_answer_raw)
        predicted_answer_cleaned = re.sub(r'\s+', ' ', predicted_answer_cleaned).strip()
        final_decoded_answers.append(predicted_answer_cleaned)
    
    return final_decoded_answers, final_predicted_numbers