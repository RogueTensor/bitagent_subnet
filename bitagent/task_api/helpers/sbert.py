from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class CachedSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.cache = {}  # Initialize an empty cache

    def encode(self, sentences, convert_to_tensor=False, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]

        results = []
        sentences_to_encode = []
        original_positions = []

        cache_key_suffix = "_tensor" if convert_to_tensor else "_array"

        for i, sentence in enumerate(sentences):
            cache_key = sentence + cache_key_suffix
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                sentences_to_encode.append(sentence)
                original_positions.append(i)
                results.append(None)  # Placeholder

        if sentences_to_encode:
            encoded = super().encode(sentences_to_encode, convert_to_tensor=convert_to_tensor, **kwargs)
            if not isinstance(encoded, list):
                encoded = [encoded[i] for i in range(len(sentences_to_encode))]
            
            for original_pos, sentence, emb in zip(original_positions, sentences_to_encode, encoded):
                cache_key = sentence + cache_key_suffix
                self.cache[cache_key] = emb
                results[original_pos] = emb

        if len(results) == 1:
            return results[0]

        if convert_to_tensor:
            return torch.stack(results)
        else:
            return np.array(results)
