# import numpy as np
import pandas as pd
import surprise

def fill_missed(df_input, df_missed = None):
    '''
    :param df_input: df with three columns - uid, iid, values
    :param df_missed: df with three columns - uid, iid, values, ??? where value_name is filled None or NaN
    :return:  df_predict - that contains all rows from df_missed but with filled by predicted value
    '''

    uid, iid, values = list(df_input)
    print('index: {}, columns: {}, values: {}'.format(uid, iid, values))

    # the default for surprise is 1 - 5 .So we will need to change this
    lower_value = df_input[values].min()
    upper_value = df_input[values].max()

    reader = surprise.Reader(rating_scale=(lower_value , upper_value))
    data = surprise.Dataset.load_from_df(df_input, reader)

    alg= surprise.SVDpp()
    alg.fit(data.build_full_trainset())

    if df_missed is None:
        return alg
    else:
        # df_missed['pred'] = test_data.apply(lambda row: alg.predict(uid=row['uid'], iid=row['iid']).est, axis=1)
        if len(list(df_missed))==2:
            df_missed[values]= 0
        missed_pred = alg.test(df_missed.values)  # Note: this accepts array in this order : uid, iid , value
        df_pred = pd.DataFrame(missed_pred)[['uid', 'iid', 'est']]
        df_pred.columns = [uid, iid, '{}_pred'.format(values)]

        return df_pred



if __name__ == '__main__':
    def get_data():
        df_filmtrust = pd.read_csv('/Users/new/science/studies/otsebriy/conductor_tools/cats/1895_estimate_msv_for_pmi_keywords/filmtrust/ratings.txt', sep=' ', names=['uid', 'iid', 'rating'])
        print('len(df_filmtrust)= {:,}'.format(len(df_filmtrust)))
        # df_filmtrust=df_filmtrust[['iid','uid','rating']]
        print(df_filmtrust.head(30))
        return df_filmtrust

    df_filmtrust = get_data()
    test_data = pd.DataFrame([
        [1, 1],
        [1, 2],
        [2, 1],
        [1, 10],
        [4, 25],
        [60, 40]], columns=['uid', 'iid'])


    df_pred = fill_missed(df_filmtrust, test_data)
    print (df_pred)
    # df_pred.to_csv('pred_filmtrust_190801_surprise.csv')
    # print (df_pred.head(100))