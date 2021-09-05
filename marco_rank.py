import tensorflow as tf
import pickle
import numpy as np
import heapq

flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "doc_emb_file", None,
    "document embedding file.")

flags.DEFINE_string(
    "query_emb_file", None,
    "query embedding file.")

flags.DEFINE_string(
    "output_file", None,
    "output file.")

flags.DEFINE_integer("batch_size", 1024, "batch size.")

flags.DEFINE_integer("topk", 1000, "topk.")


def read_embedding(emb_file):
    with open(emb_file, 'rb') as f:
        try:
            while True:
                ex = pickle.load(f)
                yield ex
        except EOFError:
            raise StopIteration


def read_query_embedding(emb_file):
    query_ids = []
    query_emb = []
    for ex in read_embedding(emb_file):
        query_ids.append(ex['query_guid'])
        query_emb.append(ex['query_embedding'])
    query_emb = np.array(query_emb)
    return query_ids, query_emb


def read_doc_embedding(emb_file):
    doc_ids = []
    doc_emb = []
    for ex in read_embedding(emb_file):
        if len(doc_ids) < FLAGS.batch_size:
            doc_emb.append(ex['doc_embedding'])
            doc_ids.append(ex['doc_guid'])
        else:
            doc_emb = np.array(doc_emb)
            doc_ids = np.array(doc_ids)
            yield doc_ids, doc_emb
            doc_emb = []
            doc_ids = []
    if len(doc_ids) > 0:
        doc_emb = np.array(doc_emb)
        doc_ids = np.array(doc_ids)
        yield doc_ids, doc_emb


def rank_doc(query_emb_file, doc_emb_files):
    all_top_doc_scores = None
    all_top_doc_ids = None
    query_ids, query_emb = read_query_embedding(query_emb_file)
    processed_doc_num = 0
    for doc_emb_file in doc_emb_files:
        for doc_ids, doc_emb in read_doc_embedding(doc_emb_file):
            doc_scores = np.matmul(query_emb, doc_emb.T)
            processed_doc_num += len(doc_emb)
            if processed_doc_num % 10*FLAGS.batch_size:
                print(f"processed {processed_doc_num} documents.")
            if not all_top_doc_ids:
                top_doc_indices = np.argsort(doc_scores, axis=-1)[:, :FLAGS.topk]
                all_top_doc_ids = doc_ids[top_doc_indices]
                all_top_doc_scores = doc_scores[top_doc_indices]
            else:
                all_top_doc_ids = np.concatenate([all_top_doc_ids, doc_ids], axis=-1)
                all_top_doc_scores = np.concatenate([all_top_doc_scores, doc_scores], axis=-1)
                top_doc_indices = np.argsort(all_top_doc_scores, axis=-1)[:, :FLAGS.topk]
                all_top_doc_ids = all_top_doc_ids[top_doc_indices]
                all_top_doc_scores = all_top_doc_scores[top_doc_indices]

    return all_top_doc_ids, query_ids


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    doc_emb_files = tf.io.gfile.glob(FLAGS.doc_emb_file)
    all_top_doc_ids, query_ids = rank_doc(FLAGS.query_emb_file, doc_emb_files)
    # output result
    with open(FLAGS.output_file, 'w') as f:
        for query_id, doc_ids in zip(query_ids, all_top_doc_ids):
            for i, doc_id in enumerate(doc_ids):
                f.write(f"{query_id}\t{doc_id}\t{i+1}\n")


if __name__ == '__main__':
    tf.app.run()