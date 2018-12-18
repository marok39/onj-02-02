import pandas as pd

def inter_rater_agreement(df):
    df1 = df.loc[df['Glenn.s.rating'] == df['Amber.s.rating']]
    df2 = df.loc[df['Glenn.s.rating'] == df['Final.rating']]
    df3 = df.loc[df['Amber.s.rating'] == df['Final.rating']]

    total_number_of_answers = len(df.index)
    same_rating = len(df1.index)
    same_rating_percentage = same_rating*100/total_number_of_answers

    glenn_final = len(df2.index)
    glenn_final_percentage = glenn_final*100/total_number_of_answers
    amber_final = len(df3.index)
    amber_final_percentage = amber_final*100/total_number_of_answers

    print('Number of answers:', total_number_of_answers)
    print('Number of answers Glenn and Amber gave the same rating: ', same_rating, ' (', "%.2f" % same_rating_percentage, '%)', sep='')
    print('Number of answers Glenn gave the final rating: ', glenn_final, ' (', "%.2f" % glenn_final_percentage, '%)', sep='')
    print('Number of answers Amber gave the final rating: ', amber_final, ' (', "%.2f" % amber_final_percentage, '%)', sep='')


if __name__ == '__main__':
    data = pd.read_csv('./input/Weightless_dataset_train.csv')
    inter_rater_agreement(data)
