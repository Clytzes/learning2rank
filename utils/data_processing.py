import numpy as np
import pandas as pd
import six
import tensorflow as tf

def load_libsvm_data(path,list_size,list_size_min = 10,num_features=None,ispredict = False):
  if not ispredict:
    return load_train_evel_libsvm_data(path, list_size,list_size_min,num_features)
  else:
    return load_predict_libsvm_data(path, list_size,list_size_min,num_features)

def load_train_evel_libsvm_data(path, list_size,list_size_min,num_features):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.compat.v1.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of queries.
  feature_map = {str(k): [] for k in np.arange(1,num_features+1)}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:

    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only. change to keep the highest 'list_size' docs only

      if doc_idx >= list_size:
        discarded_docs += 1
        continue

      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not found in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label


  tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.compat.v1.logging.info(
      "Number of documents in total: {}".format(total_docs))
  tf.compat.v1.logging.info(
      "Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
  return feature_map, np.array(label_list),qid_to_index

def load_predict_libsvm_data(path, list_size,list_size_min,num_features):
  """Returns features and labels in numpy.array."""

  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.compat.v1.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of queries.
  feature_map = {str(k): [] for k in np.arange(1,num_features+1)}
  label_list = []
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:
    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0
        for k in feature_map:
          feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
        label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1
      # Keep the first 'list_size' docs only.
      if doc_idx >= list_size:
        discarded_docs += 1
        continue
      for k, v in six.iteritems(features):
        assert k in feature_map, "Key {} not found in features.".format(k)
        feature_map[k][batch_idx][doc_idx, 0] = v
      label_list[batch_idx][doc_idx] = label

  tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
  tf.compat.v1.logging.info(
      "Number of documents in total: {}".format(total_docs))
  tf.compat.v1.logging.info(
      "Number of documents discarded: {}".format(discarded_docs))

  # Convert everything to np.array.
  for k in feature_map:
    feature_map[k] = np.array(feature_map[k])
  return feature_map, np.array(label_list),qid_to_index

def load_dataframe_data(path=None,data = None,label = 'target',list_size=1000,list_size_min = 10,mask_col=['qid','target','d'],
                        context_features=None,example_features = None,sparse_features=None,itemsname='',ispredict = False):
  if not ispredict:
    return load_train_evel_dataframe(path,data,label,list_size,list_size_min,mask_col,context_features,example_features,sparse_features)
  else:
    return load_predict_dateframe(path,data,itemsname,mask_col,context_features,example_features,sparse_features)

def load_train_evel_dataframe(path=None,data = None,label = 'target',list_size=1000,list_size_min = 10,mask_col = ['qid','target','d'],
                        context_features=None,example_features = None,sparse_features=None):

  """Returns features and labels in numpy.array."""

  assert (path is not None) or (data is not None), 'path & data should not be neither None. '
  data = pd.read_csv(path) if data is None else data


  data = data.astype({col: np.float32 if type == 'float64' else type for col, type in data.dtypes.items()})
  # data = data.astype({col: np.float64 if type == 'float32' else type for col, type in data.dtypes.items()})
  tf.compat.v1.logging.info("Loading data from {}".format(path if path is not None else 'dataframe'))

  _LABEL = label

  if example_features is None and context_features is None:
    columns = list(data.columns)
    for col in mask_col:
      columns.remove(col)
    context_features = []

  elif example_features is None and context_features is not None:
    columns = list(data.columns)
    context_features = []
    for col in mask_col + context_features:
      columns.remove(col)
  else:
    columns = example_features

  group_items = data.groupby('qid').count().iloc[:, 0].reset_index()
  group_items.columns = ['qid', 'item_num']
  group_items = group_items[group_items['item_num'] >= list_size_min]

  maxlengthonequery = group_items['item_num'].max()

  total_queries = len(data['qid'].unique())
  total_items = data.shape[0]

  final_queries = group_items.shape[0]
  discarded_qids = len(data['qid'].unique()) - group_items.shape[0]

  # The 0-based index assigned to a query.
  qid_to_index = {f'{qid}': i for i, qid in enumerate(group_items['qid'].unique())}

  # The number of docs seen so far for a query.
  # Keep the first 'list_size' docs only.
  qid_to_ndoc = {row['qid']: row['item_num'] if row['item_num'] < list_size else list_size for i, row in
                 group_items.iterrows()}

  # Each feature is mapped an array with [num_queries, list_size, 1].
  # Each  context_feature is mapped an array with [num_queries, 1].
  # Label has a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of queries.

  discarded_items = 0
  label_array = np.ones([final_queries, list_size], dtype=np.int64) * -1.

  feature_map = {str(k): np.empty([final_queries, list_size, 1], dtype=data.dtypes[k]) for k in columns}
  if context_features is not None:
    context_map = {str(k): np.empty([final_queries, 1], dtype=data.dtypes[k]) for k in context_features}

  def getgroupfeatures(group, k):
    group_item_num = group[k].shape[0]
    if group_item_num > list_size:
      fea = np.expand_dims(np.array(group[k][:list_size].tolist()), 1)
      return fea
    elif group_item_num < list_size_min:
      return
    else:
      fill = 0
      if isinstance(group[k].tolist()[0], str):
        fill = '-1'
      else:
        fill = 0.
      fea = np.expand_dims(np.array(group[k].tolist() + [fill] * (list_size - group_item_num)), 1)
      return fea

  def getcontextfeatures(group, k):
    group_item_num = group[k].shape[0]
    if group_item_num >= list_size_min:
      return group[k].tolist()[0]
    else:
      return

  def getgrouplabels(group):

    if _LABEL is None:
      return None, 0

    group_item_num = group[_LABEL].shape[0]
    if group_item_num > list_size:
      labels = np.array(group[_LABEL][:list_size].tolist())
      discarded_itemnum = group_item_num - list_size
      return labels, discarded_itemnum
    elif group_item_num < list_size_min:
      discarded_itemnum = group_item_num
      return None, discarded_itemnum
    else:
      labels = np.array(group[_LABEL].tolist() + [-1.] * (list_size - group_item_num))
      # discarded_itemnum = list_size - group_item_num
      return labels, 0

  for i, group in data.groupby('qid'):
    qid = group['qid'].tolist()[0]

    for k in feature_map:
      fea = getgroupfeatures(group, k)
      if fea is not None:
        qid_idx = qid_to_index[str(qid)]
        feature_map[k][qid_idx] = fea

    if context_features is not None:
      for k in context_map:
        con_fea = getcontextfeatures(group, k)
        if con_fea is not None:
          qid_idx = qid_to_index[str(qid)]
          context_map[k][qid_idx] = con_fea

    labels, discarded_itemnum = getgrouplabels(group)
    discarded_items += discarded_itemnum
    if labels is not None:
      qid_idx = qid_to_index[str(qid)]
      label_array[qid_idx] = labels

  tf.compat.v1.logging.info("Number of queries: {}".format(total_queries))
  tf.compat.v1.logging.info(
    "Discarg qid when qid_items < {} , Number of queries discarded: {}".format(list_size_min, discarded_qids))
  tf.compat.v1.logging.info(
    "Number of items in total: {}".format(total_items))
  tf.compat.v1.logging.info(
    "Discarg items when qid_items > {} , Number of items discarded: {}".format(list_size, discarded_items))

  tf.compat.v1.logging.info(
    "Max length in one query: {}".format(maxlengthonequery))

  label_array = label_array.astype('float32')
  if sparse_features is not None:
    for col in sparse_features:
      feature_map[col] = feature_map[col].astype('int64')

  return feature_map, label_array, qid_to_index


def load_predict_dateframe(path=None,data = None,itemsname = '',mask_col=['target','d'],context_features=None,example_features = None,sparse_features=None):
  assert (path is not None) or (data is not None), 'path & data should not be neither None. '
  data = pd.read_csv(path) if data is None else data
  data = data.astype({col: np.float32 if type == 'float64' else type for col, type in data.dtypes.items()})
  # data = data.astype({col: np.float64 if type == 'float32' else type for col, type in data.dtypes.items()})

  tf.compat.v1.logging.info("Loading data from {}".format(path if path is not None else 'dataframe'))

  if example_features is None and context_features is None:
    columns = list(data.columns)
    for col in mask_col:
      columns.remove(col)
    context_features = []

  elif example_features is None and context_features is not None:
    columns = list(data.columns)
    context_features = []
    for col in mask_col + context_features:
      columns.remove(col)
  else:
    columns = example_features


  feature_map = {str(k): np.empty([data.shape[0], 1, 1], dtype=data.dtypes[k]) for k in columns}

  for col in columns:
    feature_map[col][:,0,0] = data[col].to_list()

  if sparse_features is not None:
    for col in sparse_features:
      feature_map[col] = feature_map[col].astype('int64')

  if itemsname != '':
    return feature_map,data
  else:
    return feature_map


def dataframe_transfer_examples(inputs):

  def example_func(row):
    feature = {}
    for col in row.columns:
      value = row[col].values[0]
      if row[col].dtype == 'float32' or row[col].dtype == 'float64':
        feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
      elif row[col].dtype == 'int32' or row[col].dtype == 'int64':
        feature[col] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
      elif row[col].dtype == 'str':
        feature[col] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    example = tf.train.Example(
      features=tf.train.Features(
        feature=feature
      )
    )
    return example

  examples = []
  for i ,group in inputs.groupby(inputs.index):
    example = example_func(group)
    examples.append(example.SerializeToString())  # .SerializeToString()

  return examples


def load_examples_from_dataframe(path=None,data = None,savedatafame = True,features = None):
  assert (path is not None) or (data is not None), 'path & data should not be neither None. '
  data = pd.read_csv(path) if data is None else data
  data = data.reset_index(drop=True)
  # data = data.iloc[:10,:]
  data = data.astype({col: np.float32 if type == 'float64' else type for col, type in data.dtypes.items()})
  # data = data.astype({col: np.float64 if type == 'float32' else type for col, type in data.dtypes.items()})
  tf.compat.v1.logging.info("Loading data from {}".format(path if path is not None else 'dataframe'))
  columns = features if features is not None else data.columns
  examples = dataframe_transfer_examples(data[columns])

  if savedatafame is True:
    return examples,data
  else:
    return examples # List

def load_libsvm_convert_to_dataframe(path,num_features):
  def _parse_line(line):
    """Parses a single line in LibSVM format."""
    tokens = line.split("#")[0].split()
    assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
    label = float(tokens[0])
    qid = tokens[1]
    kv_pairs = [kv.split(":") for kv in tokens[2:]]
    features = {k: float(v) for (k, v) in kv_pairs}
    return qid, features, label

  tf.compat.v1.logging.info("Loading data from {}".format(path))

  # The 0-based index assigned to a query.
  qid_to_index = {}
  # The number of docs seen so far for a query.
  qid_to_ndoc = {}
  # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
  # a shape of [num_queries, list_size]. We use list for each of them due to the
  # unknown number of queries.
  feature_map = {f'col{str(k)}': [] for k in np.arange(1,num_features+1)}
  label_list = []
  qid_list =[]
  total_docs = 0
  discarded_docs = 0
  with open(path, "rt") as f:

    for line in f:
      qid, features, label = _parse_line(line)
      if qid not in qid_to_index:
        # Create index and allocate space for a new query.
        qid_to_index[qid] = len(qid_to_index)
        qid_to_ndoc[qid] = 0

      for k in feature_map:
        col_inx = k.strip('col')
        feature_map[k].append(features[col_inx])


      label_list.append(label)
      qid_list.append(qid)

      total_docs += 1
      batch_idx = qid_to_index[qid]
      doc_idx = qid_to_ndoc[qid]
      qid_to_ndoc[qid] += 1

      # for k, v in six.iteritems(features):
      #   assert k in feature_map, "Key {} not found in features.".format(k)
      #   feature_map[k][batch_idx][doc_idx, 0] = v
      # label_list[batch_idx][doc_idx] = label


  feature = pd.DataFrame(feature_map)
  feature['qid'] = qid_list
  feature['label'] = label_list
  feature = feature.sort_values(by = ['label','qid'],ascending=False).reset_index(drop=True)

  return feature
