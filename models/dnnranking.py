from .baseranking import BaseRankingModel

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"
DNN_SCOPE_NAME = 'dnn'


class DNNRankModel(BaseRankingModel):
    def __init__(self, params, training=True):
        super(DNNRankModel, self).__init__(params, training)
        # self.l2_reg_cross = params['l2_reg_cross']
        self.l2_reg_cross = 0.01
