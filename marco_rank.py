import tensorflow as tf
import pickle
import numpy as np
import sys


def init_flags():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Launch Distributed Training")
    parser.add_argument("--doc_emb_file", type=str, default=None)
    parser.add_argument("--query_emb_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=1000)
    return parser


def read_embedding(emb_file):
    with tf.io.gfile.GFile(emb_file, 'rb') as f:
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


def read_doc_embedding(emb_file, batch_size):
    doc_ids = []
    doc_emb = []
    for ex in read_embedding(emb_file):
        if len(doc_ids) < batch_size:
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


def rank_doc(query_emb_file, doc_emb_files, flags):
    all_top_doc_scores = None
    all_top_doc_ids = None
    query_ids, query_emb = read_query_embedding(query_emb_file)
    processed_doc_num = 0
    for doc_emb_file in doc_emb_files:
        import pdb
        pdb.set_trace()
        for doc_ids, doc_emb in read_doc_embedding(doc_emb_file, flags.batch_size):
            doc_scores = np.matmul(query_emb, doc_emb.T)
            processed_doc_num += len(doc_emb)
            tf.logging.info(f"processed {processed_doc_num} documents.")
            if all_top_doc_ids is None:
                # sort by descend order
                top_doc_indices = np.argpartition(-doc_scores, flags.topk, axis=-1)[:, :flags.topk]
                all_top_doc_ids = doc_ids[top_doc_indices]
                all_top_doc_scores = doc_scores[np.arange(len(doc_scores))[:, np.newaxis], top_doc_indices]
            else:
                all_top_doc_ids = np.concatenate([all_top_doc_ids,
                                                  np.tile(doc_ids, (len(all_top_doc_ids), 1))], axis=-1)
                all_top_doc_scores = np.concatenate([all_top_doc_scores, doc_scores], axis=-1)
                # sort by descend order
                top_doc_indices = np.argpartition(-all_top_doc_scores, flags.topk, axis=-1)[:, :flags.topk]
                all_top_doc_ids = all_top_doc_ids[np.arange(len(all_top_doc_ids))[:, np.newaxis], top_doc_indices]
                all_top_doc_scores = all_top_doc_scores[np.arange(len(all_top_doc_scores))[:, np.newaxis], top_doc_indices]

    top_doc_indices = np.argsort(-all_top_doc_scores, axis=-1)
    all_top_doc_ids = all_top_doc_ids[np.arange(len(all_top_doc_ids))[:, np.newaxis], top_doc_indices]

    return all_top_doc_ids, query_ids


def main(argv):
    parser = init_flags()
    flags = parser.parse_args(argv[1:])

    input_file = tf.io.gfile.glob(flags.query_emb_dir)[0]
    doc_emb_files = tf.io.gfile.glob(flags.doc_emb_file)
    all_top_doc_ids, query_ids = rank_doc(input_file, doc_emb_files, flags)
    output_file = flags.output_dir + "rank.test.txt"

    with tf.io.gfile.GFile(output_file, 'w') as f:
        for query_id, doc_ids in zip(query_ids, all_top_doc_ids):
            for i, doc_id in enumerate(doc_ids):
                f.write(f"{query_id}\t{doc_id}\t{i+1}\n")


if __name__ == '__main__':
    main(sys.argv)