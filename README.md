# VecDB
> A very simple library to store, retrieve, compare, update and delete embedding vectors.

It is very common in machine learning applications the need to store a large quantity of embedding vectors and compare them using functions such as cosine similary and euclidean distance. This tools helps you do that by using the H5DF format.

---
## Usage
Inside the H5DF file, the data can be segregated into dataset. This way, you can add categories to your embeddings, such as multiple albums for a face recognition system, for example. If you don't need this, don't specify anything and a custom dataset will be used.

### Initialize
The file containing the data can be created using the class constructor. If the file already exists, it will just load it.

```python
from vecdb import VecDB

# Load or create h5 file
db = VecDB(emb_dim=512, filepath='data.h5')
```

### Store new data
To add new entries to our database, we can use the method `store`. This method returns the index of the last inserted entry. This is useful to reference the embedding vector in a separate database, such as SQL or NoSQL.

```python
# One or more vectors
embeddings = np.random.randn(100000, 512)

# Store in the main dataset
last_index = db.store(embeddings)

# Store in a specific dataset
last_index = db.store(embeddings, 'my_dataset')
```

### Update entries
To update a specific embedding vector, just use the method `update`.

```python
# New embedding vector
new_embedding = np.random.randn(512)

# Update the vector with index 100 in the main dataset
db.update(100, new_embedding)

# Update the vector with index 100 in a specific dataset
db.update(100, new_embedding, 'my_dataset')
```

### Delete entries
Entries can be deleted with the method `delete`. You just have to specify the index of the entry to be deleted.

```python
# Delete entry in the main dataset
db.delete(200)

# Delete entry in a specific dataset
db.delete(200, 'my_dataset')
```

### Compare
In order to compare a new embedding vector with all entries in a dataset, you can use the `compare` method. This method needs a `func` argument in which is specified the function the be used to compare the entries.

```python
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distance

# Embedding to be compared
embedding = np.random.randn(512)

# Calculates the cosine similarity with all vectors in the main dataset
similar = db.most(
  embedding,
  func=cosine_similarity,
  n=3,
  desc=True
)

# Calculates the euclidean distance with all vectors in a specific dataset
similar = db.most(
  embedding,
  func=euclidean_distance,
  n=3,
  desc=True,
  dataset='my_dataset'
)
```