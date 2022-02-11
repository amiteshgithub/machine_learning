import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
import scipy.spatial
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)

def embed(input):
    return model(input)
    
sentence = 'this is an example for sentence embedding'

message_embeddings = embed([sentence])
print(f'Message: {sentence}')

print(f'Embedding size: {len(message_embeddings[0])}')


message_embedding_snippet = ', '.join(
    ((str(x)) for x in np.array(message_embeddings[0]).tolist()[:3])
    )
    
    
print(f'Embedding [{message_embedding_snippet}, ...]\n')


messages = ['how old are you', 'what is your age']

def similarity_measure(messages_):
    message_embeddings = embed(messages_)
    distance1 = scipy.spatial.distance.cdist([message_embeddings[0]], [message_embeddings[1]], 'cosine')[0]
    print(f'similarity score: {1-distance1}')
    return 1-distance1
    
similarity_measure(messages)

