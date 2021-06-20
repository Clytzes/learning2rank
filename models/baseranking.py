import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr

from train_opt import NadamOptimizer

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


class BaseRankingModel(object):
    def __init__(self, params, training=True):

        ini_params = {}
        ini_params.update(group_size=1,
                          hidden_layer_dims=["256", "128", "64"],
                          dropout_rate=0.5,
                          loss=None,
                          secondary_loss=None,
                          lambda_ndcg=False,
                          secondary_loss_weight=1,
                          emb_dims=8,
                          smooth_fraction=0,
                          )
        ini_params.update(params)

        self.params = ini_params
        self.group_size = self.params["group_size"]
        self.loss = self.params["loss"]
        self.hidden_layer_dims = self.params["hidden_layer_dims"]
        self.dropout_rate = self.params["dropout_rate"]
        self.use_lambda_ndcg = self.params["lambda_ndcg"]
        self.secondary_loss = self.params["secondary_loss"]
        self.secondary_loss_weight = self.params["secondary_loss_weight"]

        self.emb_dims = self.params["emb_dims"]
        self.smooth_fraction = self.params["smooth_fraction"]
        self.model_fn = self._build_model()

    def _build_model(self):
        ranking_head = self.make_ranking_head()
        model_fn = tfr.model.make_groupwise_ranking_fn(
            group_score_fn=self.make_score_fn(),
            group_size=self.group_size,
            transform_fn=self.make_transform_fn(),
            ranking_head=ranking_head
        )

        return model_fn

    def _use_multi_head(self):
        """Returns True if using multi-head."""
        return self.secondary_loss is not None

    def _lambda_weight_fn(self):
        if self.use_lambda_ndcg:
            return tfr.losses.create_ndcg_lambda_weight(smooth_fraction=self.smooth_fraction)
        else:
            return None

    # def context_feature_columns(self):
    #     return

    def example_feature_columns(self):
        """Returns the example feature columns."""
        example_feature = {}
        feature_names = self.example_features
        if self.sparse_features is None:
            example_feature = {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
                               for name in feature_names}
        else:
            for name in feature_names:
                if name not in self.sparse_features:
                    example_feature.update(
                        {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)})
                else:
                    sparse_column = tf.feature_column.categorical_column_with_identity(name, 8)
                    sparse_embedding_column = tf.feature_column.embedding_column(
                        sparse_column, self.emb_dims)
                    example_feature.update({name: sparse_embedding_column})
        return example_feature

    def make_serving_input_fn(self, ):
        """Returns serving input fn to receive tf.Example."""
        feature_spec = tf.feature_column.make_parse_example_spec(
            self.example_feature_columns().values())
        return tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)

    def _transform_fn(self, features, mode):
        """Defines transform_fn."""
        if mode == tf.estimator.ModeKeys.PREDICT:
            # We expect tf.Example as input during serving. In this case, group_size
            # must be set to 1.
            # if self.group_size != 1:
            #     raise ValueError(
            #         "group_size should be 1 to be able to export model, but get %s" %
            #         self.group_size)

            context_features, example_features = (
                tfr.feature.encode_listwise_features(
                    features=features,
                    context_feature_columns=None,
                    example_feature_columns=self.example_feature_columns(),
                    mode=mode,
                    scope="transform_layer"))
        else:
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=None,
                example_feature_columns=self.example_feature_columns(),
                mode=mode,
                scope="transform_layer")
        return context_features, example_features

    def make_transform_fn(self, ):
        """Returns a transform_fn that converts features to dense Tensors."""
        return self._transform_fn

    def make_ranking_head(self):

        if self._use_multi_head():
            primary_head = tfr.head.create_ranking_head(
                loss_fn=tfr.losses.make_loss_fn(self.loss, lambda_weight=self._lambda_weight_fn()),
                eval_metric_fns=self.get_eval_metric_fns(),
                train_op_fn=self._train_op_fn,
                name=_PRIMARY_HEAD)
            secondary_head = tfr.head.create_ranking_head(
                loss_fn=tfr.losses.make_loss_fn(self.secondary_loss, lambda_weight=self._lambda_weight_fn()),
                eval_metric_fns=self.get_eval_metric_fns(),
                train_op_fn=self._train_op_fn,
                name=_SECONDARY_HEAD)
            ranking_head = tfr.head.create_multi_ranking_head(
                [primary_head, secondary_head], [1.0, self.secondary_loss_weight])
        else:
            ranking_head = tfr.head.create_ranking_head(
                loss_fn=tfr.losses.make_loss_fn(self.loss),
                eval_metric_fns=self.get_eval_metric_fns(),
                train_op_fn=self._train_op_fn,
                name=_PRIMARY_HEAD)
        return ranking_head

    def _score_fn(self, unused_context_features, group_features, mode, unused_params, unused_config):
        """Defines the network to score a group of documents."""
        with tf.compat.v1.name_scope("input_layer"):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(self.example_feature_columns())
            ]

            # if self.sparse_features:
            #     self.sparse_emb_inputlist = [
            #         tf.compat.v1.layers.flatten(group_features[name])
            #         for name in self.sparse_features
            #     ]

            self.group_input = group_input
            input_layer = tf.concat(self.group_input, 1)
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
        self.logits = logits

        if self._use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
        else:
            return logits

    def make_score_fn(self):
        """Returns a groupwise score fn to build `EstimatorSpec`."""
        return self._score_fn

    def _train_op_fn(self, loss):
        with tf.name_scope("optimization"):
            if self.optimizer_type == "nadam":
                optimizer = NadamOptimizer(learning_rate=self.learning_rate, beta1=self.params["beta1"],
                                           beta2=self.params["beta2"], epsilon=1e-8,
                                           schedule_decay=self.params["schedule_decay"])
            elif self.optimizer_type == "adam":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                             beta1=self.params["beta1"],
                                                             beta2=self.params["beta2"], epsilon=1e-8)

            elif self.optimizer_type == "adagrad":
                optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate)

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            minimize_op = optimizer.minimize(
                loss=loss, global_step=tf.compat.v1.train.get_global_step())
            train_op = tf.group([minimize_op, update_ops])
        return train_op

    def get_train_inputs(self, features, labels, batch_size):
        """Set up training input in batches."""
        iterator_initializer_hook = IteratorInitializerHook()
        use_multi_head = self._use_multi_head()

        def _train_input_fn():
            """Defines training input fn."""
            features_placeholder = {
                k: tf.compat.v1.placeholder(v.dtype, v.shape)
                for k, v in six.iteritems(features)
            }
            if use_multi_head:
                placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
                labels_placeholder = {
                    _PRIMARY_HEAD: placeholder,
                    _SECONDARY_HEAD: placeholder,
                }
            else:
                labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)

            dataset = tf.data.Dataset.from_tensor_slices(
                (features_placeholder, labels_placeholder))
            dataset = dataset.shuffle(1000).repeat().batch(batch_size)
            iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

            if use_multi_head:
                feed_dict = {
                    labels_placeholder[head_name]: labels
                    for head_name in labels_placeholder
                }
            else:
                feed_dict = {labels_placeholder: labels}

            feed_dict.update(
                {features_placeholder[k]: features[k] for k in features_placeholder})
            iterator_initializer_hook.iterator_initializer_fn = (
                lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
            return iterator.get_next()

        return _train_input_fn, iterator_initializer_hook

    def get_eval_inputs(self, features, labels):
        """Set up eval inputs in a single batch."""
        iterator_initializer_hook = IteratorInitializerHook()
        use_multi_head = self._use_multi_head()

        def _eval_input_fn():
            """Defines eval input fn."""
            features_placeholder = {
                k: tf.compat.v1.placeholder(v.dtype, v.shape)
                for k, v in six.iteritems(features)
            }
            if use_multi_head:
                placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
                labels_placeholder = {
                    _PRIMARY_HEAD: placeholder,
                    _SECONDARY_HEAD: placeholder,
                }
            else:
                labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            dataset = tf.data.Dataset.from_tensors(
                (features_placeholder, labels_placeholder))
            iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            if use_multi_head:
                feed_dict = {
                    labels_placeholder[head_name]: labels
                    for head_name in labels_placeholder
                }
            else:
                feed_dict = {labels_placeholder: labels}

            feed_dict.update(
                {features_placeholder[k]: features[k] for k in features_placeholder})
            iterator_initializer_hook.iterator_initializer_fn = (
                lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
            return iterator.get_next()

        return _eval_input_fn, iterator_initializer_hook

    def get_predict_inputs(self, features):
        """Set up eval inputs in a single batch."""
        iterator_initializer_hook = IteratorInitializerHook()

        def _predict_input_fn():
            """Defines eval input fn."""
            features_placeholder = {
                k: tf.compat.v1.placeholder(v.dtype, v.shape)
                for k, v in six.iteritems(features)
            }

            dataset = tf.data.Dataset.from_tensors(features_placeholder)
            iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
            feed_dict = {features_placeholder[k]: features[k] for k in features_placeholder}
            iterator_initializer_hook.iterator_initializer_fn = (
                lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
            return iterator.get_next()

        return _predict_input_fn, iterator_initializer_hook

    def fit(self, train_data, valid_data
            , output_dir, batch_size
            , learning_rate
            , num_train_steps
            , optimizer_type
            , linear_optimizer=None
            , dnn_optimizer=None
            , sparse_features=None
            , save_checkpoints_steps=100
            , save_summary_steps=100
            , early_stopping=True
            , max_steps_without_decrease=100
            , run_every_steps=100
            , exporter='best'
            ):

        features, labels = train_data
        features_vali, labels_vali = valid_data

        self.featurescolumns = list(features.keys())
        self.features_dtypes = {key: features[key].dtype for key in features.keys()}

        self.example_features = list(features.keys())
        self.context_features = []
        self.sparse_features = sparse_features

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        # self.linear_optimizer = linear_optimizer if linear_optimizer is not None else optimizer_type
        # self.dnn_optimizer = dnn_optimizer if dnn_optimizer is not None else optimizer_type

        self.save_checkpoints_steps = save_checkpoints_steps
        self.save_summary_steps = save_summary_steps
        self.max_steps_without_decrease = max_steps_without_decrease
        self.run_every_steps = run_every_steps

        self.exporter = exporter

        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=tf.estimator.RunConfig(self.output_dir
                                          , save_checkpoints_steps=self.save_checkpoints_steps
                                          , save_summary_steps=self.save_summary_steps
                                          )
        )

        train_input_fn, train_hook = self.get_train_inputs(features, labels, batch_size=self.batch_size)
        vali_input_fn, vali_hook = self.get_eval_inputs(features_vali, labels_vali)

        if early_stopping:
            early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
                estimator=self.estimator, metric_name="loss",
                max_steps_without_decrease=self.max_steps_without_decrease, eval_dir=None, min_steps=0,
                run_every_secs=None, run_every_steps=self.run_every_steps
            )

            train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                hooks=[train_hook, early_stopping_hook],
                max_steps=self.num_train_steps)
        else:
            train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                hooks=[train_hook],
                max_steps=self.num_train_steps)

        if self.exporter == 'best':
            exporter = tf.estimator.BestExporter(
                name="best_exporter",
                serving_input_receiver_fn=self.make_serving_input_fn())
        elif self.exporter == 'latest':
            exporter = tf.estimator.LatestExporter(
                "latest_exporter",
                serving_input_receiver_fn=self.make_serving_input_fn()
            )

        # Export model to accept tf.Example when group_size = 1.
        if self.group_size == 1:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                exporters=exporter,
                start_delay_secs=0,
                throttle_secs=30)
        else:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                exporters=exporter,
                start_delay_secs=0,
                throttle_secs=30)

        # Train and validate
        return tf.estimator.train_and_evaluate(self.estimator, train_spec, vali_spec)

    def predict(self, features, model_dir=None):
        """
            https://github.com/tensorflow/ranking/issues/69
            https://github.com/tensorflow/ranking/issues/186
            https://github.com/tensorflow/ranking/issues/211
        """
        if model_dir != None:
            checkpoint_path = tf.train.latest_checkpoint(model_dir)
        else:
            checkpoint_path = None

        predict_input_fn, predict_hook = self.get_predict_inputs(features)
        predict_data = self.estimator.predict(input_fn=predict_input_fn, hooks=[predict_hook],
                                              checkpoint_path=checkpoint_path)

        if self._use_multi_head():
            result = [pre[_PRIMARY_HEAD] for pre in list(predict_data)]
            return np.concatenate(result)
        else:
            # result = [pre[_PRIMARY_HEAD] for pre in list(predict_data)]
            # return np.concatenate(result)

            return np.array([pre[0] for pre in list(predict_data)])

    def get_eval_metric_fns(self, ):
        """Returns a dict from name to metric functions."""
        metric_fns = {}
        metric_fns.update({
            "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
                tfr.metrics.RankingMetricKey.ARP,
                tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
            ]
        })
        metric_fns.update({
            "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
                tfr.metrics.RankingMetricKey.NDCG, topn=topn)
            for topn in [1, 3, 5, 10, 20]
        })
        return metric_fns

    def valid_evaluate(self, vali_input_fn, vali_hook):
        # Export model to accept tf.Example when group_size = 1.
        if self.group_size == 1:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                exporters=tf.estimator.LatestExporter(
                    "latest_exporter",
                    serving_input_receiver_fn=self.make_serving_input_fn()),
                start_delay_secs=0,
                throttle_secs=30)
        else:
            vali_spec = tf.estimator.EvalSpec(
                input_fn=vali_input_fn,
                hooks=[vali_hook],
                steps=1,
                start_delay_secs=0,
                throttle_secs=30)
        return vali_spec

    def evaluate(self, features_test, labels_test):
        # Evaluate on the test data.
        test_input_fn, test_hook = self.get_eval_inputs(features_test, labels_test)
        return self.estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])

    def export_saved_model(self, save_mode_dir):
        model_dir = self.estimator.export_saved_model(export_dir_base=save_mode_dir,
                                                      serving_input_receiver_fn=self.make_serving_input_fn())
        return model_dir

    def load_savedmodel_predict(self, model_dir):
        imported = tf.saved_model.load(model_dir)
        return imported

    def predict_from_example(self, model_dir, examples):
        imported = self.load_savedmodel_predict(model_dir)

        pre = imported.signatures["predict"](
            examples=tf.constant(examples)
        )
        if self._use_multi_head():
            return pre[f'{_PRIMARY_HEAD}/output']
        else:
            return
