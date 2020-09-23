import numpy as np
from vecdb import VecDB
from flask import Flask, request, json, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
db = SQLAlchemy(app)

vdb = VecDB(filepath='data.h5', emb_dim=512)

# Face Model
class Face(db.Model):
  id = db.Column(db.Integer, primary_key=True)
  name = db.Column(db.String(50))
  embedding_key = db.Column(db.Integer)
  created_at = db.Column(db.DateTime, default=datetime.now)

# Fake ML model
def model():
  return np.random.randn(512)

# Routes
@app.route('/store', methods=['POST'])
def store():
  name = request.form.get('name')
  pred = model()
  emb_key = vdb.store(pred)

  face = Face(name=name, embedding_key=emb_key)
  db.session.add(face)
  db.session.commit()

  return jsonify({ "status": "ok", "msg": "New face added" })

@app.route('/most_similar', methods=['POST'])
def most_similar():
  emb = model()
  key = vdb.most(emb, func=cosine_similarity)
  face = Face.query.filter_by(embedding_key=int(key)).first()

  if face is not None:
    return jsonify({ "name": face.name })
  return jsonify({ "status": "error", "msg": "Could not compare faces" })

if __name__ == '__main__':
  app.run()