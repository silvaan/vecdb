import numpy as np
from vecdb import VecDB
from sklearn.metrics.pairwise import cosine_similarity

# Load or create h5 file
db = VecDB(emb_dim=512, filepath='data.h5')

# Store data
embeddings = np.random.randn(100000, 512)
db.store(embeddings)

# Update entry
new_embedding = np.random.randn(512)
db.update(100, new_embedding)

# Delete entry
db.delete(200)

# Get most similar
embedding = np.random.randn(512)
similar = db.most(embedding, func=cosine_similarity, n=3, desc=True)
print(similar)