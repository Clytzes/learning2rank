"""
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

from deepctr.layers.interaction import FM

from .baseranking import *

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


class DeepFMRankModel(BaseRankingModel):
    def __init__(self, params, training=True):
        super(DeepFMRankModel, self).__init__(params, training)

    def example_feature_columns(self):
        """Returns the example feature columns."""

        example_feature = {}
        feature_names = self.example_features

        self.dnn_feature_columns = []
        self.linear_feature_columns = []

        if self.sparse_features is None:
            example_feature = {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                               for name in feature_names}
            self.linear_feature_columns.append(list(example_feature.values()))

        else:

            for name in feature_names:
                if name not in self.sparse_features:
                    numeric_column = tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                    example_feature.update({name: numeric_column})
                    self.linear_feature_columns.append(numeric_column)
                else:
                    sparse_column = tf.feature_column.categorical_column_with_identity(name, 8)
                    sparse_embedding_column = tf.feature_column.embedding_column(
                        sparse_column, self.emb_dims)
                    example_feature.update({name: sparse_embedding_column})

                    self.linear_feature_columns.append([sparse_column])

        self.dnn_feature_columns.append(list(example_feature.values()))
        return example_feature

    def _score_fn(self, unused_context_features, group_features, mode, unused_params, unused_config):
        """Defines the network to score a group of documents."""

        # self.linear_feature_columns,self.dnn_feature_columns = self.get_linear_dnn_features()

        with tf.compat.v1.name_scope("input_layer"):
            self.group_features = group_features
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(self.example_feature_columns())
            ]

            self.sparse_emb_inputlist = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in self.sparse_features
            ]

            sparse_emb = [tf.expand_dims(fea, 1) for fea in self.sparse_emb_inputlist]
            sparse_emb = tf.concat(sparse_emb, axis=1)
            self.sparse_emb = sparse_emb
            fm_logit = FM()(sparse_emb)

            self.group_input = group_input
            input_layer = tf.concat(self.group_input, 1)
            print('看这里')
            print(input_layer)
            print(self.sparse_emb)
            tf.compat.v1.summary.scalar("input_sparsity",
                                        tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar("input_max",
                                        tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar("input_min",
                                        tf.reduce_min(input_tensor=input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        cur_layer = tf.compat.v1.layers.batch_normalization(
            input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in self.hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                        tf.nn.zero_fraction(cur_layer))

        cur_layer = tf.compat.v1.layers.dropout(
            cur_layer, rate=self.dropout_rate, training=is_training)

        logits = tf.compat.v1.layers.dense(cur_layer, units=self.group_size)
        self.logits = logits + fm_logit

        if self._use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: self.logits, _SECONDARY_HEAD: self.logits}
        else:
            return self.logits
