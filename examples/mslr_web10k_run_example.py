from models import DNNRankModel
from utils import *


def tain_predict_output():
    train_result = model.fit(train_data=(features, labels)
                             , valid_data=(features_vali, labels_vali)
                             , sparse_features=sparse_features
                             , output_dir=output_dir
                             , batch_size=64
                             , learning_rate=0.01
                             , num_train_steps=5000
                             , optimizer_type='adagrad'  # adam
                             , save_checkpoints_steps=100
                             , early_stopping=True
                             , max_steps_without_decrease=400
                             , run_every_steps=100
                             )
    eval_result = model.evaluate(features_test, labels_test)
    return train_result, eval_result


losses = ['approx_ndcg_loss']
secondary_losses = [None]
lambda_ndcg_list = [False]
group_size_list = [1]
folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
all_results_df = pd.DataFrame({})
all_results_ls = []

for split_fold in folds:
    for cat in ['train','vali','test']:
        train_path = f'data/MSLR-WEB10K/{split_fold}/{cat}.txt'
        data = load_libsvm_convert_to_dataframe(train_path,num_features=136)
        data.to_csv(f'data/MSLR-WEB10K/{split_fold}/{cat}.csv',index=False)

for split_fold in folds:
    train_path = f'data/MSLR-WEB10K/{split_fold}/train.csv'
    vali_path = f'data/MSLR-WEB10K/{split_fold}/vali.csv'
    test_path = f'data/MSLR-WEB10K/{split_fold}/test.csv'

    sparse_features = None
    features, labels, qid = load_dataframe_data(train_path, list_size=200, list_size_min=5,
                                                sparse_features=sparse_features, label='label',
                                                mask_col=['qid', 'label'])
    features_vali, labels_vali, qid = load_dataframe_data(vali_path, list_size=200, list_size_min=5,
                                                          sparse_features=sparse_features, label='label',
                                                          mask_col=['qid', 'label'])
    features_test, labels_test, qid = load_dataframe_data(test_path, list_size=200, list_size_min=5,
                                                          sparse_features=sparse_features, label='label',
                                                          mask_col=['qid', 'label'])

    for group_size in group_size_list:
        for i, loss in enumerate(losses):
            for j, secondary_loss in enumerate(secondary_losses):
                for lambda_ndcg in lambda_ndcg_list:
                    modelname = 'dnn' + '_' + f'loss{i + 1}' + '_' + (
                        'nosecloss' if secondary_loss is None else f'secloss{j + 1}') + '_' + \
                                ('lamweight' if lambda_ndcg is True else 'nolamweight') + '_' + \
                                f'groupsize{group_size}'

                    print(f'{modelname} is training...')
                    output_dir = 'output/MSLR/cpt/{}'.format(modelname + '_' + f'{split_fold}')

                    params = {}
                    params.update(group_size=group_size,
                                  hidden_layer_dims=["256", "128", "64"],
                                  dropout_rate=0.5,
                                  loss=loss,
                                  secondary_loss=secondary_loss,
                                  lambda_ndcg=lambda_ndcg,
                                  secondary_loss_weight=1,
                                  emb_dims=8
                                  )

                    model = DNNRankModel(params)
                    train_result, evaluate_result = tain_predict_output()

                    train_mctric = {}
                    eval_mctric = {}
                    appn_info = {}
                    if not train_mctric:
                        train_mctric = {f'train/{key}': [train_result[0][key]] for key in train_result[0]}
                        eval_mctric = {f'test/{key}': [evaluate_result[key]] for key in evaluate_result}
                        appn_info = {'model': [modelname], 'fold': [split_fold], 'group_size': group_size, \
                                     'loss': loss, 'secondary_loss': secondary_loss,
                                     'lambda_ndcg': 1 if lambda_ndcg is True else 0}
                    else:
                        [train_mctric[f'train/{key}'].append(train_result[0][key]) for key in train_result[0]]
                        [eval_mctric[f'test/{key}'].append(evaluate_result[key]) for key in evaluate_result]
                        appn_info['model'].append(model)
                        appn_info['fold'].append(split_fold)
                        appn_info['group_size'].append(group_size)
                        appn_info['loss'].append(loss)
                        appn_info['secondary_loss'].append(secondary_loss)
                        appn_info['lambda_ndcg'].append(lambda_ndcg)

                    all_results = {}
                    for result in [train_mctric, eval_mctric, appn_info]:
                        all_results.update(result)
                    _all_results = pd.DataFrame(all_results)
                    all_results_df = all_results_df.append(_all_results)

all_results_df.to_csv(f'output/MSLR/report/MSLR_{modelname}.csv')
