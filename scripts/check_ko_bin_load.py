import os
from gensim.models import Word2Vec, KeyedVectors


paths = []
for root, dirs, files in os.walk("."):
    if "ko.bin" in files:
        paths.append(os.path.join(root, "ko.bin"))

print("paths:", paths)

for path in paths:
    print("try:", path)

    try:
        model = Word2Vec.load(path)
        print("Word2Vec.load ok", len(model.wv.key_to_index), model.vector_size)
    except Exception as e:
        print("Word2Vec.load fail", type(e).__name__, str(e)[:200])

    try:
        kv = KeyedVectors.load(path)
        print("KeyedVectors.load ok", len(kv.key_to_index), kv.vector_size)
    except Exception as e:
        print("KeyedVectors.load fail", type(e).__name__, str(e)[:200])

    try:
        kv = KeyedVectors.load_word2vec_format(path, binary=True)
        print("load_word2vec_format binary ok", len(kv.key_to_index), kv.vector_size)
    except Exception as e:
        print("load_word2vec_format binary fail", type(e).__name__, str(e)[:200])

    try:
        kv = KeyedVectors.load_word2vec_format(path, binary=False)
        print("load_word2vec_format text ok", len(kv.key_to_index), kv.vector_size)
    except Exception as e:
        print("load_word2vec_format text fail", type(e).__name__, str(e)[:200])
