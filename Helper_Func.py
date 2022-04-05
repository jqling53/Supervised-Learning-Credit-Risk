#Function for cleaning the data
#Inputs:
#    df: a pandas dataframe
# Output: a pandas dataframe

def CleanData(data):
    
    import pandas as pd
    import numpy as np
    
    #Valid responses
    data = data[(data['loan_status'] == 'Fully Paid') | (data['loan_status'] == 'Charged Off')]

    #Drop column if more than alpha are non-Null
    alpha = 0.90

    data_cleaned = data.dropna(axis='columns',thresh = np.ceil(alpha*len(data)))
    data_cleaned = data_cleaned.dropna()

    #Split into numeric and nonnumeric
    numeric = data_cleaned[data_cleaned.columns[np.where(data_cleaned[data_cleaned.columns].dtypes=='float64')]]
    nonnumeric = data_cleaned[data_cleaned.columns[np.where(data_cleaned[data_cleaned.columns].dtypes!='float64')]]

    #Drop columns we dont want
    nndrops = ["id","grade","sub_grade","emp_title","loan_status","zip_code","addr_state",
               "title","pymnt_plan","url","hardship_flag","issue_d","last_pymnt_d","earliest_cr_line",
              "last_credit_pull_d","debt_settlement_flag", 'disbursement_method']
    ndrops = ["installment","int_rate","last_pymnt_amnt","out_prncp","out_prncp_inv",
 "total_pymnt","total_pymnt_inv","total_rec_int","total_rec_late_fee","total_rec_prncp",
              'out_prncp', 'out_prncp_inv', 'policy_code',"recoveries","collection_recovery_fee",
             'funded_amnt','funded_amnt_inv',"last_pymnt_amnt",'last_fico_range_low']
    high_corr = ['fico_range_high',
             'num_sats',
             'num_rev_tl_bal_gt_0',
             'tot_hi_cred_lim',
             'total_il_high_credit_limit',
             'bc_util',
             'bc_open_to_buy',
             'avg_cur_bal',
             'num_rev_accts',
             'num_bc_sats',
             'last_fico_range_high',
             'num_op_rev_tl',
             'num_actv_bc_tl',
             'total_rev_hi_lim',
             'num_tl_30dpd',
             'num_tl_op_past_12m',
             'percent_bc_gt_75',
             'tax_liens',
             'total_acc',
             'num_actv_rev_tl',
             'num_tl_90g_dpd_24m',
             'pub_rec_bankruptcies',
             'mo_sin_rcnt_rev_tl_op'
             ]
    numeric = numeric.drop(ndrops,axis = 1)
    numeric = numeric.drop(high_corr,axis = 1)
    nonnumeric = nonnumeric.drop(nndrops,axis = 1)
    
    #Numeric
    #Function to transform skewed data into a more normal distribution by applying a power (default cube root)
    def to_power(x, power = 1/3, trans_positive = True):
        """
        :param x: input data in the form of a dataframe column
        :param power: the power to raise the data to, or 'log'
        :param positive: flag if the data needs to be transformed to positive
        :return: powered: transformed dataframe column
        """
        if trans_positive == True:
            min_val = min(x)
            x = [(z - min_val + 1) for z in x]  
        if power == 'log':
            powered = [np.log(z) for z in x]
        else:
            powered = [z**power for z in x]
        return(powered)
    
    #Which transformation to use for each column
    log_cols = [1,2,3,5,6,7,8,10,11,12,14,15,16,19,20,21,22,23,24,25,26,27]
    sqrroot_cols = [0]
    cuberoot_cols = [4,9,13,17,18]
    
    #Apply transformation to numeric variables
    numeric_cols = list(numeric.columns)
    numeric_cols.remove('pct_tl_nvr_dlq')
    for i in log_cols:
        numeric[numeric_cols[i]] = to_power(numeric[numeric_cols[i]], power = 'log', trans_positive = True)
    for i in sqrroot_cols:
        numeric[numeric_cols[i]] = to_power(numeric[numeric_cols[i]], power = 1/2, trans_positive = True)
    for i in cuberoot_cols:
        numeric[numeric_cols[i]] = to_power(numeric[numeric_cols[i]], power = 1/3, trans_positive = True)     
    
    #Normalize numeric
    numeric=(numeric-numeric.mean())/numeric.std()

    #Categorical 
    def ohe_columns(df, columns):
        """
        :param df: input data in the form of a dataframe
        :param columns: list of columns to encode
        :return: df: dataframe with the encoded columns
        """
        for col in columns:
            df = pd.concat([df.drop([col], axis=1),
                            pd.get_dummies(df[col],
                            drop_first=True, prefix=col)], axis=1)
        return df

    nonnumeric = ohe_columns(nonnumeric,nonnumeric.columns)
    

    #Response Variable
    response=ohe_columns(data_cleaned[["loan_status"]],["loan_status"])
    new_cols = ['loan_status']
    response.columns = new_cols
    response = (response - 1)/255
    
    #Rename Problimatic Column
    nonnumeric = nonnumeric.rename(columns={"emp_length_< 1 year":"emp_length_lessthan 1 year"})

    #Concat
    output = pd.concat([numeric,nonnumeric,response],axis=1)
    return(output) 

#Function for spliting data into three pieces
#Inputs:
#    data: a pandas dataframe
#    vprop: proportion of validation set
#    tprop: proportion of test set
#    seed: seed for replicability
# Output: a list of pandas dataframes


def DataSplit(data,vprop,tprop,seed=None):
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    #Split proportions
    p1 = tprop+vprop
    p2 = tprop/p1
    
    #Calling split functions
    y = data["loan_status"]
    X = data.drop(columns=["loan_status"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p1,random_state=seed)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=p2,random_state=seed)

    #Returning DataFrames
    data_train = pd.concat([X_train,y_train],axis=1)
    data_valid = pd.concat([X_valid,y_valid],axis=1)
    data_test = pd.concat([X_test,y_test],axis=1)
    return([data_train,data_valid,data_test])

#Function for spliting data into three pieces
#Inputs:
#    y_predict: Predicted values by model
#    y_valid: Validation Outcomes
# Output: thresholds and optimal id

def ROC_cutoff(y_predict,y_valid,silence=False):

    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    # Compute ROC curve and ROC area for each class

    fpr, tpr, thresholds = roc_curve(y_valid,y_predict)
    roc_auc = auc(fpr, tpr)
    
    if not silence:
        #Plotting the LR ROC Curve
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()

    #Computing the best threshold
    gmeans = tpr * (1-fpr)
    id = np.argmax(gmeans)
    if not silence:
        print("Optimal Threshold:",thresholds[id])
    return([thresholds,id])
