from utilities.importFile import *
from skrebate import ReliefF

def dataCollection(data):
    """
    Processes data collection techniques from the dataset and formats them appropriately.
    Args:
        data (DataFrame): The DataFrame containing data collection technologies and associated metrics.  
    Returns:
        DataFrame: The processed DataFrame with data collection techniques and metrics.
    """
    processed_data = []
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        technologies = row['data_collection_technology']
        if pd.notna(technologies):
            tech_list = str(technologies).strip('[]').split(',')
            for tech in tech_list:
                tech = tech.strip().strip("'")
                # Append processed data to the list
                processed_data.append([tech, row['performance_mean'], row['DataSize_all'], row['metrics_recorded'], row['augmentation_used'], row['augmentation_techniques'], row['color_norm'], row['balanced'], row['balanced_techniques'],row['multiple_cohorts'], row['external_val_set'], row['pre_training'], row['datatype_pretrained']])
    # Convert processed data into a DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['dataColl_techniques', 'performance_mean', 'DataSize_all', 'metrics_recorded', 'augmentation_used', 'augmentation_techniques', 'color_norm', 'balanced', 'balanced_techniques', 'multiple_cohorts', 'external_val_set', 'pre_training', 'datatype_pretrained'])
    # Replace numerical codes with descriptive names for data collection techniques
    df_processed.replace({"dataColl_techniques": data_collection_technique_dict}, inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    return df_processed

def dataCollection_Augmentation(data, final_features):
    """
    Extends the data frame with details about augmentation techniques used.
    Args:
        data (DataFrame): The DataFrame to process.
        final_features (list): List of final features used for augmentation.
    Returns:
        tuple: The updated DataFrame and list of final features including new augmentation details.
    """
    processed_data = []
    # Load augmentation techniques from Excel file
    aug_file = pd.read_excel('data_augmentation.xlsx')
    aug_file['Type'] = aug_file['Type'].str.lower()
    aug_types = aug_file['Type'].tolist()
    aug_types = [n.split(', ') for n in aug_types]
    new_column_names = ['aug_color','aug_geometrics', 'aug_deletion', 'aug_kernels', 'aug_others', 'aug_synthetic', 'aug_none']
    aug_dict = dict(zip(new_column_names, aug_types))
    final_features.extend(new_column_names)
    df_processed = pd.DataFrame(processed_data, columns=final_features)
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        aug = row['augmentation_techniques']
        added_result = [0, 0, 0, 0, 0, 0, 0]
        if pd.notna(aug):
            aug_list = str(aug).strip('[]').split(', ')
            for aug_i in aug_list:
                aug_i = aug_i.lower()
                aug_i = aug_i.strip("'")
                result = row[:].tolist()
                key_i = find_key_for_word(aug_dict, aug_i)
                added_result = [a or b for a, b in zip(added_result, [key_i == s for s in new_column_names])]
                result.extend(added_result)
                df_processed.loc[index] = result
        else:
            result = row[:].tolist()
            result.extend([False, False, False, False, False, False, True])
            df_processed.loc[index] = result
    return df_processed, final_features

def dataCollection_datasize(data, final_features):
    """
    Creates new binary columns in the DataFrame based on data size criteria.
    Args:
        data (DataFrame): The DataFrame to process.
        final_features (list): List of final features including data size details.  
    Returns:
        tuple: The updated DataFrame and list of final features including data size details.
    """
    data['DataSize_500'] = data['DataSize_all'] < 500
    data['DataSize_500>'] = data['DataSize_all'] > 500
    final_features.extend(['DataSize_500', 'DataSize_500>'])
    return data, final_features

def dataCollection_pretraining(data, final_features):
    """
    Extends the DataFrame with pre-training data types used as features.
    Args:
        data (DataFrame): The DataFrame to process.
        final_features (list): List of final features to be updated with pre-training data types. 
    Returns:
        tuple: The updated DataFrame and list of final features including pre-training details.
    """
    processed_data = []
    data['datatype_pretrained'] = data['datatype_pretrained'].fillna("['none']")
    new_column_names = data['datatype_pretrained'].tolist()
    new_column_names = [n.strip('[]') for n in new_column_names]
    new_column_names = [n.split(", ") for n in new_column_names]
    flattened_list = []
    for sublist in new_column_names:
        flattened_list.extend(sublist)
    new_column_names = np.unique(flattened_list)
    new_column_names = [n.strip("'") for n in new_column_names]
    final_features.extend(new_column_names)
    df_processed = pd.DataFrame(processed_data, columns=final_features)
    for index, row in data.iterrows():
        pretrained = row['datatype_pretrained']
        added_result = [0 for _ in range(len(new_column_names))]
        if pd.notna(pretrained):
            aug_list = str(pretrained).strip('[]').split(', ')
            for aug_i in aug_list:
                aug_i = aug_i.strip("'")
                result = row[:].tolist()
                added_result = [a or b for a, b in zip(added_result, [aug_i == s for s in new_column_names])]
                result.extend(added_result)
                df_processed.loc[index] = result
        else:
            result = row[:].tolist()
            result.extend([False for _ in range(len(new_column_names)) - 1])
            result.append(True)
            df_processed.loc[index] = result
    df_processed.drop(columns=['datatype_pretrained'], axis=1, inplace=True)
    final_features = [f for f in final_features if f != 'datatype_pretrained']
    return df_processed, final_features

def clean_dataset(df):
    """
    Cleans the dataset by removing rows with NaN values in specific columns.
    Args:
        df (DataFrame): The DataFrame to clean.   
    Returns:
        DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.dropna(subset=['DataSize_all', 'performance_mean', 'metrics_recorded'])
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def relief_analysis(relief_data):
    """
    Performs a ReliefF feature importance analysis on the provided data.
    Args:
        relief_data (DataFrame): The DataFrame containing the data for feature importance analysis.
    Returns:
        dict: A dictionary with features as keys and their importance scores as values.
    """
    relief_data.drop(columns=['dataColl_techniques', 'metrics_recorded'], inplace=True)
    features = [c for c in relief_data.columns.tolist() if (c != 'performance_mean') and (c != 'dataColl_techniques') and (c != 'metrics_recorded')]
    r = ReliefF(n_neighbors=3)
    feature_importance = dict(zip(features, r.feature_importances_))
    return feature_importance

def main_process(network_type='ResNet', cancer_type='pancancer', data_coll='H&E', clinical_task='all', ml_task='all', general_cal=False):
    """
    Executes the main processing pipeline for filtering and processing data based on specific parameters and conditions.
    This function loads and filters the data according to network type, cancer type, data collection method, clinical task, 
    and machine learning task specified. It then preprocesses the data by managing different aspects like data augmentation, 
    datasize categorization, and pre-training data handling, followed by a cleanup process. Finally, it conducts a ReliefF 
    analysis to determine feature importance

    Args:
        network_type (str): Specifies the type of network to filter by (e.g., 'ResNet', 'ensemble', 'sequential').
        cancer_type (str): Specifies the cancer type to filter by, with 'pancancer' indicating all types.
        data_coll (str): Specifies the data collection technique to filter by, with 'H&E' as an example.
        clinical_task (str): Specifies the clinical task to filter by, with 'all' indicating no specific filter.
        ml_task (str): Specifies the machine learning task to filter by, with 'all' indicating no specific filter.
        general_cal (bool): If True, performs a general ReliefF feature importance analysis across all data.

    Returns:
        None: This function does not return a value but may output files containing feature importance data or 
              print statements indicating the status of data processing depending on the parameters and data conditions.
    """
    # Load data
    data = data_frame.copy()
    # Include only relevant data based on parameters
    if cancer_type != 'pancancer':
        data = data[data[cancer_type] == True]
    if clinical_task != 'all':
        data = data[data[clinical_task] == True]
    if ml_task != 'all':
        data = data[data['ml_task_description'] == ml_task]
    if network_type == 'single':
        data = data[(data['ensemble_model'] == False) & (data['sequential_model'] == False)]
    elif network_type == 'ensemble':
        data[data['ensemble_model'] == True]
    elif network_type =='sequential':
        data = data[data['sequential_model'] == True]
    # Subset features and target column
    features_clean = ['DataSize_all','data_collection_technology', 'metrics_recorded', 'augmentation_used', 'augmentation_techniques', 'color_norm', 'balanced', 'balanced_techniques', 'multiple_cohorts', 'external_val_set', 'pre_training', 'datatype_pretrained']
    target_column = 'performance_mean'
    data = data[features_clean + [target_column]]
    data = dataCollection(data)
    cleaned_data = clean_dataset(data)
    if data_coll == 'H&E':
        cleaned_data = cleaned_data[cleaned_data['dataColl_techniques'] == 'H&E']
    # Transform augmentation techniques column
    cleaned_data, final_features = dataCollection_Augmentation(cleaned_data, features_clean)
    # Transform DataSize_all column
    cleaned_data, final_features = dataCollection_datasize(cleaned_data, final_features)
    # Transform pre_training column
    cleaned_data, final_features = dataCollection_pretraining(cleaned_data, final_features)
    # Remove unnecessary features
    cleaned_data.drop(columns=['augmentation_techniques', 'aug_none', 'balanced_techniques', 'Unspecified', 'none', 'bal_none', 'bal_boostrapping', 'bal_unknown', 'Medical', 'bal_ensemble', 'unknown', 'DataSize_all'], axis=1, inplace=True, errors='ignore')
    final_features = [f for f in final_features if f not in ['augmentation_techniques', 'aug_none', 'balanced_techniques', 'Unspecified', 'none', 'bal_none', 'bal_boostrapping', 'bal_unknown', 'Medical', 'bal_ensemble', 'unknown', 'DataSize_all']]
    # Process metrics_recorded column
    cleaned_data = process_network(cleaned_data)
    cleaned_data.iloc[:, 4:] = cleaned_data.iloc[:, 4:].astype('int64')
    cleaned_data.to_csv('cleaned_data.csv')
    if general_cal:
        fi_general = relief_analysis(cleaned_data)
        res = pd.DataFrame.from_dict(fi_general, orient='index')
        res = res.reset_index()
        res = res.rename(columns={'index': 'Feature names', 0: 'Feature importance - general'})
        res.to_csv(f'FI_Dashboard/FI_general.csv') 
    else:            
        fi_general = pd.read_csv('FI_Dashboard/FI_general.csv')
        if network_type == 'single' or network_type == 'ensemble' or network_type == 'sequential':
            if cleaned_data.shape[0] < 10:
                print(f'Two few or no data for combination {network_type}-{cancer_type}-{data_coll}-{clinical_task}-{ml_task}')
                return
            feature_importance = relief_analysis(cleaned_data)
            res = pd.DataFrame.from_dict(feature_importance, orient='index')
            res = res.reset_index()
            res = res.rename(columns={'index': 'Feature names', 0: 'Feature importance'})
            res = pd.merge(res, fi_general, on='Feature names', how='left')
            res.drop(columns=['Unnamed: 0'], inplace=True)
            res['Feature importance - general'] = fi_general['Feature importance - general']
            res.to_csv(f'FI_Dashboard/FI-{network_type}-{cancer_type}-{data_coll}-{clinical_task}-{ml_task}.csv') 
        else:
            for network, df in cleaned_data.groupby('metrics_recorded'):
                if network == network_type:
                    if df.shape[0] < 10:
                        print(f'Two few or no data for combination {network}-{cancer_type}-{data_coll}-{clinical_task}-{ml_task}')
                        return
                    feature_importance = relief_analysis(df)
                    feature_importances[network] = feature_importance
                    res = pd.DataFrame.from_dict(feature_importance, orient='index')
                    res.to_csv(f'FI_networks_final/FI_{network}_pancancer_H&E.csv') 

