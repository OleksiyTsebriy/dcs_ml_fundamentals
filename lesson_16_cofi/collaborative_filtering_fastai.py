# Some sources:
# https://towardsdatascience.com/collaborative-filtering-with-fastai-3dbdd4ef4f00
# https://becominghuman.ai/collaborative-filtering-using-fastai-a2ec5a2a4049
# https://medium.com/jovian-io/hows-that-movie-neural-collaborative-filtering-with-fastai-64d573f4b2dc

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use('TkAgg')

from fastai.collab import CollabDataBunch, collab_learner

def predict_set(dl, learn, input_columns):
    uid, iid, values = input_columns
    df_pred = pd.DataFrame()
    for batch in iter(dl):
        (uids, iids), _ = batch
        preds = learn.model(uids, iids)
        preds = np.array(preds.data).squeeze() # in case NN using it returns 2 -dimensional array

        uids = np.array(uids.data)
        iids = np.array(iids.data)

        df = pd.DataFrame({uid: uids, iid: iids, '{}_pred'.format(values): preds})
        df_pred = df_pred.append(df, ignore_index=True)

    df_pred = df_pred.reset_index(drop=True)

    return df_pred

def fill_missed(df_input, df_missed = None, use_nn= False,  n_cycles= 10, lr = 1e-1 , n_ft_cycles= 10, lr_ft = 1e-5 ):
    '''
    :param df_input: df with three columns - uid, iid, values
    :param df_missed: df with three columns - uid, iid, values, ??? where value_name is filled None or NaN
    :return:  df_predict - that contains all rows from df_missed but with filled by predicted value
    '''

    uid, iid, values = list(df_input)
    print('index: {}, columns: {}, values: {}'.format(uid, iid, values))

    # create data bunch
    if df_missed is None:
        df_test= df_filmtrust
    else:
        df_test = df_missed
    data_bunch = CollabDataBunch.from_df(df_input, seed=42, valid_pct=0.1,
                                   user_name=uid,
                                   item_name=iid,
                                   rating_name=values,
                                   test=df_test)

    # create learner
    rating_min, rating_max = df_input[values].min(), df_input[values].max()


    if use_nn:
        print ('Using EmbeddingNN Model')
        learn = collab_learner(data_bunch, use_nn=True, emb_szs={uid: 40, iid:40}, layers=[256, 128], y_range=(rating_min, rating_max))
    else:
        print ('Using EmbeddingDotBias Model')
        learn = collab_learner(data_bunch, n_factors=20, y_range=(rating_min, rating_max), wd=1e-1)

    # train
    print ('\nTraining model...')
    learn.fit_one_cycle(n_cycles, lr)

    print ('\nFine tuning...')
    learn.fit_one_cycle(n_ft_cycles, lr_ft)

    # predict
    df_test_pred= predict_set(data_bunch.test_dl, learn, list(df_input))

    # evaluate
    df_train_pred = predict_set(data_bunch.train_dl, learn, list(df_input))
    df_merged = df_train_pred.merge(df_input, how='left', on=([uid, iid])).dropna()
    score= r2_score(df_merged [values], df_merged ['{}_pred'.format(values)])
    print ('\nR2 score = {}'.format( score))

    # merge to test set
    df_merged = df_test_pred.merge(df_input, how= 'left', on=([uid, iid]))

    return df_merged




if __name__ == '__main__':
    def get_data():
        df_filmtrust = pd.read_csv('/Users/new/science/studies/otsebriy/conductor_tools/cats/1895_estimate_msv_for_pmi_keywords/filmtrust/ratings.txt', sep=' ', names=['uid', 'iid', 'rating'])
        print('len(df_filmtrust)= {:,}'.format(len(df_filmtrust)))
        print ('First samples:')
        print(df_filmtrust.head(10))
        return df_filmtrust

    df_filmtrust = get_data()
    test_data = pd.DataFrame([
        [1, 1],
        [1, 2],
        [2, 1],
        [1, 10],
        [4, 25],
        [60, 40]], columns=['uid', 'iid'])


    df_pred = fill_missed(df_filmtrust, test_data, n_cycles=2, n_ft_cycles= 2, use_nn=False)

    print('\nResults:')
    print (df_pred)



