import os
import numpy as np
import h5py

DEFAULT_DATASET = 'main'

class VecDB():
  def __init__(self, emb_dim, filepath='data.h5'):
    """Constructor.

    Args:
        emb_dim: The size of the embeddings vectors.
        filepath: The path of the H5 file.

    Returns:
        The return value. True for success, False otherwise.

    """
    self.filepath = filepath
    self.emb_dim = emb_dim
    
    if not os.path.isfile(filepath):
      hf = h5py.File(self.filepath, 'a')
      hf.close()

  def store(self, embeddings, dataset=DEFAULT_DATASET):
    """Stores a single ou multiple new entries.

    Args:
        embeddings: The embeddings vectors.
        dataset: The dataset/category of the entry.

    Returns:
        The index of the last added entry.

    """
    data = embeddings.reshape(-1, self.emb_dim)
    with h5py.File(self.filepath, 'r+') as fh:
      if dataset not in fh.keys():
        fh.create_dataset(dataset, data=data, maxshape=(None,self.emb_dim))
      else:
        fh[dataset].resize((fh[dataset].shape[0]+1,fh[dataset].shape[1]))
        fh[dataset][-1,:] = data
      last_id = len(fh[dataset])-1
    return last_id

  def get(self, index=None, dataset=DEFAULT_DATASET):
    """Returns a single ou multiple entries.

    Args:
        index: The embedding index.
        dataset: The dataset/category of the entry.

    Returns:
        An array containing the embeddings.

    """
    with h5py.File(self.filepath, 'r') as fh:
      if index != None:
        return fh[dataset][index,:]
      return fh[dataset][:]

  def update(self, index, embedding, dataset=DEFAULT_DATASET):
    """Updates a single entry.

    Args:
        index: The embedding index.
        embedding: The new embedding vector.
        dataset: The dataset/category of the entry.

    """
    with h5py.File(self.filepath, 'a') as fh:
      fh[dataset][index,:] = embedding

  def delete(self, index, dataset=DEFAULT_DATASET):
    """Deletes a single entry.

    Args:
        index: The embedding index.
        dataset: The dataset/category of the entry.

    """
    with h5py.File(self.filepath, 'a') as fh:
      fh[dataset][index,:] = np.nan

  def compare(self, embedding, func, desc=True, dataset=DEFAULT_DATASET):
    """Compares a given embedding vector with the entire dataset.

    Args:
        embedding: The embedding vector to be compared.
        func: A function for comparing.
        desc: Descending order
        dataset: The dataset/category of the entry.

    Returns:
        A dict containing the comparing values(similarity, distance...) for each
        entry in a dataset.

    """
    with h5py.File(self.filepath, 'r') as fh:
      data = np.array(fh[dataset])

      is_nan = np.isnan(fh[dataset]).any(axis=1)
      where_nan = np.where(is_nan)[0]
      data[is_nan] = 0

      values = func(embedding.reshape(-1, self.emb_dim), data).reshape(-1)

    values_dict = {k:v for k,v in enumerate(values) if k not in where_nan}
    sorted_values = {k:v for k,v in sorted(values_dict.items(), key=lambda item : item[1], reverse=desc)}
    
    return sorted_values

  def most(self, embedding, func, n=1, desc=True, dataset=DEFAULT_DATASET):
    """Compares a given embedding vector with the entire dataset.

    Args:
        embedding: The embedding vector to be compared.
        func: A function for comparing.
        n: Top n
        desc: Descending order
        dataset: The dataset/category of the entry.

    Returns:
        A list with the keys ordered accordingly to the comparing function.

    """
    values = self.compare(embedding, func, desc, dataset)
    if n == 1:
      return list(values.keys())[0]
    return list(values.keys())[:n]