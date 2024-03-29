<p align="center">
  <img src="VecDB.svg">
</p>

**VecDB** is a tool to help you to store, manage and compare embeddings vectors for machine learning applications.

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vecdb.svg)](https://pypi.python.org/pypi/vecdb/) [![PyPI license](https://img.shields.io/pypi/l/vecdb.svg)](https://pypi.python.org/pypi/vecdb/)

## Embeddings

It is very common in machine learning applications the need to store a large quantity of embedding vectors and compare them using functions such as cosine similary or euclidean distance. **VecDB** helps you to do that in a fast and efficient way by using H5DF files.

## Dependencies
- [NumPy](https://www.numpy.org/)
- [h5py](https://www.h5py.org/)

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

## License

MIT © [jrmiranda](https://github.com/jrmiranda)
