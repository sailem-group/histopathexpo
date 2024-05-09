# Import necessary modules
from utilities.importFile import *
from sklearn.neighbors import KNeighborsRegressor


def cleaning_specific_columns(df, columns):
    """
    Cleans specific columns in the DataFrame by removing rows with NaN or infinite values and converts numeric columns to float64.
    Args:
        df (DataFrame): DataFrame to be cleaned.
        columns (list): List of columns to clean.
    Returns:
        DataFrame: The cleaned DataFrame.
    """
    # Make a copy of the DataFrame to avoid modifying the original one
    df_cleaned = df.copy()

    # Iterate through specified columns
    for column in columns:
        # Check if the column exists in the DataFrame
        if column in df_cleaned.columns:
            # Remove rows with NaN, Inf, or -Inf values in the specified column
            df_cleaned = df_cleaned[df_cleaned[column].notna()]
            df_cleaned = df_cleaned[~df_cleaned[column].isin(
                [np.inf, -np.inf])]

            # Convert the column to type np.float64 if it's numeric
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                df_cleaned[column] = df_cleaned[column].astype(np.float64)
        else:
            raise ValueError(f"Column {column} not found in DataFrame")

    return df_cleaned


def processing_data_collection(data):
    """
    Processes data collection techniques from the dataset.
    Args:
        data (DataFrame): The DataFrame containing data collection techniques and associated metrics.
    Returns:
        DataFrame: The processed DataFrame.
    """
    processed_data = []

    for index, row in data.iterrows():
        # Extract data collection technologies and title from each row
        technologies = row['data_collection_technology']
        title = row['title']

        # Skip rows with missing data collection technologies
        if pd.isna(technologies):
            continue

        # Split data collection technologies and process each one
        tech_list = str(technologies).strip('[]').split(',')
        for tech in tech_list:
            tech = tech.strip().strip("'")
            processed_data.append([tech, row['performance_mean'], title,
                                   row['multiple_cohorts'], row['external_val_set'], row['ml_task_description'],
                                   row['task_seg+obj_det'],  row['task_detection'],   row['task_prognosis'],
                                   row['task_survival'],    row['task_treatment_design'],    row['task_risk_prediction'],
                                   row['task_diagnosis_subtyping'], row['task_others'], row['subspec_bladder'],
                                   row['subspec_brainca'], row['subspec_breastca'], row['subspec_cervical'], row['subspec_colorectal'],
                                   row['subspec_dermca'], row['subspec_endometrial'], row['subspec_esophagus'],
                                   row['subspec_gastric'], row['subspec_headandneck_others'], row['subspec_haemonc'], row['subspec_hepca'],
                                   row['subspec_kidney'], row['subspec_lungca'], row['subspec_metastasis'], row['subspec_oral'],
                                   row['subspec_others'], row['subspec_ovarian'], row['subspec_pancreatic'], row['subspec_prosca'],
                                   row['subspec_thyroid'], row['doi'], row['article_date'], row['network_type'], row['performance_auc'],
                                   row['performance_sensitivity (recall)'], row['performance_specificity'], row['DataSize_all'],
                                   row['class_labels'], row['raw_data_availability'], row['code_availability'], row['methodological'],
                                   row['performance_cindex'], row['performance_precision (PPV)'], row[
                'performance_NPV'], row['performance_F1'],
                row['performance_accuracy'], row['benchmarking'], row['implementation_detail'], row['data_source']])

    # Convert processed data into a DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['dataColl_techniques', 'performance_mean', 'title', 'multiple_cohorts', 'external_val_set',
                                                         'ml_task_description', 'task_seg+obj_det',  'task_detection',   'task_prognosis',
                                                         'task_survival',    'task_treatment_design',    'task_risk_prediction',     'task_diagnosis_subtyping',
                                                         'task_others', 'subspec_bladder', 'subspec_brainca', 'subspec_breastca', 'subspec_cervical', 'subspec_colorectal',
                                                         'subspec_dermca', 'subspec_endometrial', 'subspec_esophagus', 'subspec_gastric', 'subspec_headandneck_others',
                                                         'subspec_haemonc', 'subspec_hepca', 'subspec_kidney', 'subspec_lungca', 'subspec_metastasis', 'subspec_oral',
                                                         'subspec_others', 'subspec_ovarian', 'subspec_pancreatic', 'subspec_prosca', 'subspec_thyroid', 'doi', 'article_date',
                                                         'network_type', 'performance_auc', 'performance_sensitivity (recall)', 'performance_specificity', 'DataSize_all',
                                                         'class_labels', 'raw_data_availability', 'code_availability', 'methodological', 'performance_cindex',
                                                         'performance_precision (PPV)', 'performance_NPV', 'performance_F1', 'performance_accuracy', 'benchmarking',
                                                         'implementation_detail', 'data_source'])

    # Replace numerical codes with descriptive names for data collection techniques
    df_processed.replace(
        {"dataColl_techniques": data_collection_technique_dict}, inplace=True)

    return df_processed


def one_hot_encoding(data):
    """
    Performs one-hot encoding on specified categorical columns of a DataFrame.
    Args:
        data (DataFrame): The DataFrame to encode.
    Returns:
        DataFrame: The one-hot encoded DataFrame.
    """
    # Perform one-hot encoding on specified columns
    encoded_data = pd.get_dummies(
        data, columns=['ml_task_description', 'dataColl_techniques', 'data_source'])
    return encoded_data


def retrieving_cancer_types(row):
    """
    Retrieves cancer types based on the subspecialty columns in the DataFrame.
    Args:
        row (Series): A row of DataFrame.
    Returns:
        str: Comma-separated string of cancer types.
    """
    # Get columns with value 1 and replace names using the dictionary
    cancer_types = [cancer_typesdict.get(col, col)
                    for col in subspec_columns if row[col] == 1]
    # Join the column names with a comma if more than one type is found
    return ', '.join(cancer_types)

# Function for preprocessing data


def preprocessing_data():
    """
    Preprocesses the given data by cleaning, processing data collection techniques, encoding, and setting up an index.
    Args:
        data (DataFrame): The data to preprocess.
    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    # Clean specified columns in the DataFrame
    data_frame = data.copy()
    data_frame = data_frame[cleaningnames]
    data_frame = cleaning_specific_columns(data_frame, tobeClean)
    # Process data collection techniques
    data_processed = processing_data_collection(data_frame)
    data_frame = data_processed
    data_frame['DataSize_all_copy'] = data_frame['DataSize_all']
    data_frame['network_type_copy'] = data_frame['network_type']
    data_frame['multiple_cohorts_copy'] = data_frame['multiple_cohorts']
    # Define bins for binning the 'DataSize_all' column
    # Use float('inf') for an open-ended upper bound
    bins = [0, 500, 1000, 5000, float('inf')]
    labels = [1, 2, 3, 4]
    # Create a new categorical column based on data size
    data_frame['DataSize_all'] = pd.to_numeric(
        data_frame['DataSize_all'], errors='coerce')
    data_frame['DataSize_all'] = pd.cut(
        data_frame['DataSize_all'], bins=bins, labels=labels, include_lowest=True)
    # Get cancer types from subspecialty columns
    data_frame['cancer_type'] = data_frame.apply(
        retrieving_cancer_types, axis=1)
    # Replace numerical codes with descriptive names for data sources
    data_frame['data_source'] = data_frame['data_source'].replace(
        public_dataset)
    # Convert 'article_date' to datetime format
    if data_frame['article_date'].dtype != 'datetime64[ns]':
        data_frame['article_date'] = pd.to_datetime(
            data_frame['article_date'], errors='coerce')
    # Set index for the DataFrame
    data_frame.set_index(['title', 'doi', 'article_date', 'network_type_copy', 'performance_auc', 'performance_sensitivity (recall)',
                          'performance_specificity', 'cancer_type', 'DataSize_all_copy', 'raw_data_availability',
                          'code_availability', 'methodological', 'performance_cindex', 'performance_precision (PPV)',
                          'performance_NPV', 'performance_F1', 'performance_accuracy', 'benchmarking',
                          'implementation_detail', 'external_val_set', 'multiple_cohorts_copy'], inplace=True)
    # Perform one-hot encoding on the DataFrame
    encoded_data = one_hot_encoding(data_frame)
    data_frame = encoded_data

    return data_frame

def qualityIndexForData(df):
    """
    Enhances the DataFrame by adding a quality score based on several indicators.
    Only the 'q_score' column is retained alongside original dataframe columns after calculations.
    Parameters:
        df (DataFrame): The input DataFrame containing research paper metrics. 
    Returns:
        DataFrame: The modified DataFrame with an additional 'q_score' column.
    """
    # Store original columns to preserve them after adding new computed columns
    original_columns = set(df.columns)

    # Compute boolean flags based on data availability and code availability
    df['data_avai_yes'] = df['raw_data_availability'] == 'yes'
    df['code_avai'] = df['code_availability'].fillna(0).astype(bool)
    df['methodology'] = df['methodological'].fillna(0.0).astype(bool)

    # List of performance-related columns for calculating the number of non-null performance metrics
    performance_columns = [
        'performance_cindex', 'performance_auc', 'performance_precision (PPV)',
        'performance_specificity', 'performance_NPV', 'performance_sensitivity',
        'performance_F1', 'performance_accuracy'
    ]
    df['performance_metrics'] = df[performance_columns].notna().sum(axis=1) >= 3

    # Compute boolean flags for benchmarking and implementation details
    df['bench'] = df['benchmarking'].fillna(0.0).astype(bool)
    df['imple_det'] = df['implementation_detail'].fillna(0.0).astype(bool)

    # Calculate external validation flag considering multiple cohorts
    if 'multiple_cohorts_copy' in df.columns:
        df['external_val'] = df['external_val_set'].fillna(0).astype(
            bool) | df['multiple_cohorts_copy'].fillna(0).astype(bool)
    else:
        df['external_val'] = df['external_val_set'].fillna(0).astype(
            bool) | df['multiple_cohorts'].fillna(0).astype(bool)

    # Assign weights to each quality indicator and calculate the overall quality score
    weights = {
        'data_avai_yes': 1.0,
        'code_avai': 1.0,
        'methodology': 1.0,
        'performance_metrics': 1.0,
        'external_val': 1.0,
        'bench': 1.0,
        'imple_det': 1.0
    }
    df['q_score'] = sum(df[col] * weight for col, weight in weights.items())

    # Identify new columns added for calculations except 'q_score'
    added_columns = set(df.columns) - original_columns
    added_columns.remove('q_score')  # Ensure 'q_score' is retained

    # Drop all newly added columns except 'q_score'
    df.drop(columns=list(added_columns), inplace=True)

    return df

def retrieving_top_neighbors(input_data, k=15):
    """
    Retrieves top neighbors for a given input using KNN.
    Args:
        input_data (str): JSON string of input data.
        k (int): Number of neighbors to retrieve.
    Returns:
        DataFrame: DataFrame containing the top neighbors.
    """
    # Convert input data from JSON to a pandas Series
    json_dict = json.loads(input_data)
    keys_view = json_dict.keys()
    column_names = list(keys_view)
    column_names.append('performance_mean')
    input_data = pd.Series(json_dict)

    if 'highQuality' in input_data:
        highQuality = input_data['highQuality']
        del input_data['highQuality']  # Remove 'highQuality' from the data
    else:
        highQuality = 0.0

    # Preprocess data
    data_frame = preprocessing_data()
    data_frame = data_frame[column_names]
    data_frame = cleaning_specific_columns(data_frame, column_names)

    # Define special columns for weighting
    spec_columns = [
        col for col in data_frame.columns if col.startswith('subspec_')]

    # Define a custom weight function
    def custom_weights_function(distances):
        return 1 / (1 + distances)

    # Initialize KNN model with or without custom weights
    if len(spec_columns) > 0:
        knn = KNeighborsRegressor(
            n_neighbors=k, weights=custom_weights_function)
    else:
        knn = KNeighborsRegressor(n_neighbors=k)

    knn.fit(data_frame.drop(
        columns=['performance_mean']), data_frame['performance_mean'])

    # Apply weighting to the input data
    for col in spec_columns:
        if col in input_data.index:
            input_data[col] *= 9  # Adjust weight factor as needed

    # Find indices of top neighbors
    distances, indices = knn.kneighbors(input_data.values.reshape(1, -1))

    # Retrieve top neighbors
    neighbors = data_frame.iloc[indices[0]].sort_values(
        by='performance_mean', ascending=False)
    
    if highQuality != '0.0':
        top_neighbors = qualityIndexForData(neighbors)
        highQuality_value = float(highQuality)
        # Filter out records where 'q_score' is less than 4
        top_neighbors = top_neighbors[top_neighbors['q_score'] >= highQuality_value]
        return top_neighbors

    return neighbors
