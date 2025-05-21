import re
from collections import Counter, OrderedDict
import unicodedata
import torch
import torch.nn as nn
import math
import struct
import os

# --- Tokenization Patterns ---
hex_pattern_str = r'0x[0-9A-Fa-f]+'
number_pattern_str = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?'
date_pattern_str = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
time_pattern_str = r'\b\d{1,2}:\d{2}(?::\d{2})?\b'

# Regex to identify distinct tokens
tokenizer_pattern_re = re.compile(
    r'(?:@[\w]+)|'                      # @tags (e.g., @xcite, @xmath0)
    r'(?:e\.g\.)|'                      # "e.g."
    r'(?:i\.e\.)|'                      # "i.e."
    r'(?:0x[0-9a-fA-F]+)|'              # Hex numbers
    r'(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|' # Dates
    r'(?:\d{1,2}:\d{2}(?::\d{2})?)|'    # Times
    r'(?:[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)|' # General numbers
    r'(?:\.\.\.)|'                      # Ellipsis
    r'(?:->|<-|==|!=|>=|<=|>>|<<|::=|--)|' # Compound operators
    r'(?:[A-Za-z]+(?:[\'-][A-Za-z]+)*)|' # Words (with optional hyphen/apostrophe)
    r'(?:[\u2010-\u2015])|'             # Unicode hyphens
    r'(?:[.,;:!?()\[\]{}@\'\"`~&*_+=<>$#%^\\/|-])|' # Single punctuation/symbols
    r'(\s+)|'                           # Whitespace
    r'(?:\S)'                           # Any other non-whitespace character
)

def tokenize(text):
    """
    Tokenizes input text, handling special number formats and capitalization.
    Returns a list of tokens and a map for numerical token data.
    """
    raw_entries = [] # Stores (value, type, raw_string) for <|num|> tokens
    tokens = []

    text = unicodedata.normalize('NFKC', text)
    text = text.replace("−", "-")
    text = re.sub(r'(\d)\s*,\s*(\d{3})', r'\1,\2', text)
    text = re.sub(r'(\d)\s*\.\s*(\d+)', r'\1.\2', text)
    
    last_idx = 0
    for match in tokenizer_pattern_re.finditer(text):
        part_str = match.group(0)
        if not part_str: 
            continue

        if re.fullmatch(r'\s+', part_str):
            tokens.append(part_str)
            last_idx = match.end()
            continue 

        is_number_like = False
        
        if re.fullmatch(date_pattern_str, part_str):
            for comp in re.split(r'([/-])', part_str):
                if re.fullmatch(r'[/-]', comp):
                    tokens.append(comp)
                elif comp: 
                    try:
                        val_comp = float(comp)
                        tokens.append("<|num|>")
                        raw_entries.append((val_comp, 'int_date_comp', comp))
                    except ValueError: tokens.append(comp)
            is_number_like = True
        elif re.fullmatch(time_pattern_str, part_str):
            for comp in re.split(r'([:])', part_str):
                if comp == ':':
                    tokens.append(comp)
                elif comp:
                    try:
                        val_comp = float(comp)
                        tokens.append("<|num|>")
                        raw_entries.append((val_comp, 'int_time_comp', comp))
                    except ValueError: tokens.append(comp)
            is_number_like = True
        elif re.fullmatch(hex_pattern_str, part_str):
            try:
                val = float(int(part_str, 16))
                tokens.append("<|num|>")
                raw_entries.append((val, 'hex', part_str))
            except ValueError: tokens.append(part_str)
            is_number_like = True
        elif re.fullmatch(number_pattern_str, part_str):
            try:
                clean_num_str = part_str.replace(',', '')
                fv = float(clean_num_str)
                typ = 'float' if '.' in clean_num_str or 'e' in clean_num_str.lower() else 'int'
                tokens.append("<|num|>")
                raw_entries.append((fv, typ, part_str))
            except ValueError: tokens.append(part_str)
            is_number_like = True

        if is_number_like:
            last_idx = match.end() 
            continue 

        if re.fullmatch(r'[A-Za-z]+(?:[\'-][A-Za-z]+)*', part_str): 
            if part_str.isupper() and len(part_str) > 1:
                tokens.append("<|allcaps|>")
                tokens.append(part_str.lower())
            elif part_str[0].isupper() and (len(part_str) == 1 or part_str[1:].islower()):
                tokens.append("<|cap|>")
                tokens.append(part_str.lower())
            else: tokens.append(part_str)
        else:
            tokens.append(part_str)
        
        last_idx = match.end() 

    if last_idx < len(text):
        remaining_text = text[last_idx:]
        if remaining_text: tokens.append(remaining_text)

    number_map = {}
    raw_entry_idx = 0
    for token_idx, tok in enumerate(tokens):
        if tok == "<|num|>":
            if raw_entry_idx < len(raw_entries):
                number_map[token_idx] = raw_entries[raw_entry_idx]
                raw_entry_idx += 1
            else:
                number_map[token_idx] = (None, 'unknown', '<|num|>')
                print(f"Warning: <|num|> token at index {token_idx} has no corresponding raw_entry.")
                
    return tokens, number_map

def detokenize(tokens, number_map=None):
    """
    Reconstructs the original string from a list of tokens and a number map,
    applying capitalization and intelligent spacing.
    """
    output_parts = []
    cap_next_token = False
    allcaps_next_token = False

    attached_to_prev = {'.', ',', ':', ';', '!', '?', ')', ']', '}', '\'', '"', '...'} 
    attached_to_next = {'(', '[', '{', '\'', '"', '$'} 
    
    def is_at_token(token):
        return token.startswith('@x') and re.fullmatch(r'@[\w]+', token)

    for i, token_from_list in enumerate(tokens):
        current_word_to_append = ""

        if token_from_list == '<|cap|>':
            cap_next_token = True
            continue 
        if token_from_list == '<|allcaps|>':
            allcaps_next_token = True
            continue 

        if token_from_list == '<|num|>':
            if number_map and i in number_map:
                _, _, raw_number_str = number_map[i]
                current_word_to_append = raw_number_str
            else: current_word_to_append = '<|num|>'
            cap_next_token = False
            allcaps_next_token = False
        else: 
            current_word_to_append = token_from_list
            if allcaps_next_token:
                current_word_to_append = token_from_list.upper()
                allcaps_next_token = False
            elif cap_next_token:
                current_word_to_append = token_from_list.capitalize()
                cap_next_token = False
        
        if not output_parts:
            output_parts.append(current_word_to_append)
            continue

        prev_actual_token = tokens[i-1] if i > 0 else None 
        prev_output_part = output_parts[-1]

        if re.fullmatch(r'\s+', current_word_to_append):
            output_parts.append(current_word_to_append)
            continue
        
        if re.fullmatch(r'\s+', prev_output_part):
            output_parts.append(current_word_to_append)
            continue

        if current_word_to_append in attached_to_prev or \
           is_at_token(current_word_to_append) or \
           (current_word_to_append == '-' and prev_actual_token == '<|num|>'):
            output_parts.append(current_word_to_append)
            continue
        
        if prev_actual_token in attached_to_next: 
            output_parts.append(current_word_to_append)
            continue
        
        if (current_word_to_append == '-' or current_word_to_append == ':') and \
           prev_actual_token == '<|num|>' and \
           i + 1 < len(tokens) and tokens[i+1] == '<|num|>':
            output_parts.append(current_word_to_append)
            continue
            
        output_parts.append(' ')
        output_parts.append(current_word_to_append)
            
    return ''.join(output_parts)


def summarize_tokens(tokens):
    """
    Counts token frequencies and assigns unique IDs based on first appearance.
    """
    counts = Counter(tokens)
    token_ids = OrderedDict()
    idx_counter = 0
    for t in tokens:
        if t not in token_ids:
            token_ids[t] = idx_counter
            idx_counter +=1
            
    summary_list = []
    for token_str in token_ids.keys():
        summary_list.append((token_str, token_ids[token_str], counts[token_str]))
            
    return summary_list