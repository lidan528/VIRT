import tensorflow as tf
import random

def count_input_examples(file_pattern):
    if tf.gfile.IsDirectory(file_pattern):
        if file_pattern.endswith("/"):
            file_pattern = file_pattern + "*"
        else:
            file_pattern = file_pattern + "/*"
    file_list = tf.gfile.Glob(file_pattern)
    sorted_file_list = sorted(file_list)
    file_size = []
    total_size = 0
    for f in sorted_file_list:
        size = sum(1 for _ in tf.python_io.tf_record_iterator(f))
        file_size.append((f, size))
        total_size += size
        if len(file_size) % 20 == 0:
            print("Counting input files %d, total examples %d" % (len(file_size), total_size))

    return total_size, file_size

def return_tfr_path(file_pattern):
    if tf.gfile.IsDirectory(file_pattern):
        if file_pattern.endswith("/"):
            file_pattern = file_pattern + "*"
        else:
            file_pattern = file_pattern + "/*"
    file_list = tf.gfile.Glob(file_pattern)
    sorted_file_list = sorted(file_list)
    return sorted_file_list


def input_evenly_builder(total_size, file_size, total_workers, worker_index):
    examples_per_worker = total_size / total_workers
    examples_mod_left = total_size % total_workers

    examples_to_skip = examples_per_worker * worker_index
    examples_to_read = examples_per_worker

    # the first mod-th worker has one more example
    if worker_index < examples_mod_left:
        examples_to_skip += worker_index * 1
        examples_to_read += 1
    else:
        examples_to_skip += examples_mod_left * 1
    print("Worker %d need to skip %d and read %d" % (worker_index, examples_to_skip, examples_to_read))

    input_datasets = []
    for (f, size) in file_size:
        # skip the whole file
        if examples_to_skip >= size:
            examples_to_skip -= size
            continue

        # no more need to read
        if examples_to_read <= 0:
            break

        # we definitely need to read this file
        d = tf.data.TFRecordDataset(f)
        read_start = 0
        read_end = 0

        # skip partial examples if still need skip
        if examples_to_skip > 0:
            d = d.skip(examples_to_skip)
            read_start = examples_to_skip
            examples_to_skip = 0

        file_remain = size - read_start
        # take if only partial examples need to read
        if examples_to_read <= file_remain:
            d = d.take(examples_to_read)
            read_end = read_start + examples_to_read
            examples_to_read = 0
        # read all the remain
        else:
            read_end = size
            examples_to_read -= file_remain

        input_datasets.append((d, read_start, read_end, f, size))

    dataset_size = 0
    random.shuffle(input_datasets)
    for (d, start, end, f, size) in input_datasets:
        dataset_size += (end - start)
        print("Worker %d read file %s(%d) from %d to %d, total %d" % (worker_index, f, size, start, end, dataset_size))

    # concat datasets as a tree, reduce from O(N) to O(LogN)
    dataset_list_leaf = map(lambda (d, start, end, f, size): d, input_datasets)
    dataset_list_root = []
    while len(dataset_list_leaf) > 1:
        for i in range(0, len(dataset_list_leaf), 2):
            if i+1 < len(dataset_list_leaf):
                dataset = dataset_list_leaf[i].concatenate(dataset_list_leaf[i+1])
                dataset_list_root.append(dataset)
            else:
                dataset_list_root.append(dataset_list_leaf[i])
        dataset_list_leaf = dataset_list_root
        dataset_list_root = []
    final_dataset = dataset_list_leaf[0]

    # the last (total_worker - mod) workers need repeat one example, to keep same eg. counts with the first mod-th workers
    if 0 < examples_mod_left <= worker_index:
        final_dataset = final_dataset.concatenate(final_dataset.take(1))

    return final_dataset










