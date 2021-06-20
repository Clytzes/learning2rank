"""
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""
from deepctr.layers.interaction import CrossNet

from .baseranking import BaseRankingModel

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"
DNN_SCOPE_NAME = 'dnn'


class DCNRankModel(BaseRankingModel):
    def __init__(self, params, training=True):
        super(DCNRankModel, self).__init__(params, training)
        # self.l2_reg_cross = params['l2_reg_cross']
        self.l2_reg_cross = 0.01

    def get_linear_dnn_features(self):
        """Returns the example feature columns."""
        feature_names = self.example_features
        example_feature = {}
        linear_feature_columns = []
        if self.sparse_features is None:
            example_feature = {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                               for name in feature_names}
            linear_feature_columns = list(example_feature.values())
        else:
            for name in feature_names:
                if name not in self.sparse_features:
                    numeric_column = tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                    example_feature.update({name: numeric_column})
                    linear_feature_columns.append(numeric_column)
                else:
                    sparse_column = tf.feature_column.categorical_column_with_identity(name, 8)
                    sparse_embedding_column = tf.feature_column.embedding_column(
                        sparse_column, self.emb_dims)
                    example_feature.update({name: sparse_embedding_column})
                    linear_feature_columns.append(sparse_column)

        dnn_feature_columns = list(example_feature.values())
        return linear_feature_columns, dnn_feature_columns

    def get_linear_dnn_features_dict(self):
        """Returns the example feature columns."""
        feature_names = self.example_features
        dnn_feature_columns = {}
        linear_feature_columns = {}
        if self.sparse_features is None:
            dnn_feature_columns = {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                                   for name in feature_names}
            linear_feature_columns = dnn_feature_columns
        else:
            for name in feature_names:
                if name not in self.sparse_features:
                    numeric_column = tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                    dnn_feature_columns.update({name: numeric_column})
                    linear_feature_columns.update({name: numeric_column})

                else:
                    sparse_column = tf.feature_column.categorical_column_with_identity(name, 8)
                    sparse_embedding_column = tf.feature_column.embedding_column(
                        sparse_column, self.emb_dims)
                    dnn_feature_columns.update({name: sparse_embedding_column})
                    linear_feature_columns.update({name: sparse_column})

        return linear_feature_columns, dnn_feature_columns

    def _score_fn(self, unused_context_features, group_features, mode, unused_params, unused_config):

        """Defines the network to score a group of documents."""
        self.linear_feature_columns, self.dnn_feature_columns = self.get_linear_dnn_features_dict()

        with tf.compat.v1.name_scope("input_layer"):
            self.group_features = group_features

            # group_input = [
            #     tf.compat.v1.layers.flatten(group_features[name])
            #     for name in sorted(self.example_feature_columns())
            # ]

            self.dnn_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(self.dnn_feature_columns)
            ]

            # self.group_input = group_input
            input_layer = tf.concat(self.dnn_input, 1)

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
        print(cur_layer)
        self.cur_layer = cur_layer

        logits = tf.compat.v1.layers.dense(cur_layer, units=self.group_size)

        cross_out = CrossNet(2, l2_reg=self.l2_reg_cross)(input_layer)
        print(cross_out)
        self.cross_out = cross_out
        # stack_out = tf.keras.layers.Concatenate()([cross_out, deep_out])
        final_logit = tf.keras.layers.Dense(
            1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=1024))(cross_out)
        print(final_logit)

        self.logits = logits + final_logit

        if self._use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: self.logits, _SECONDARY_HEAD: self.logits}
        else:
            return self.logits
