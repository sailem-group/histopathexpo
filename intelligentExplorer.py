# Import necessary modules
from utilities.importFile import *
from sklearn.neighbors import NearestNeighbors


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
        technologies = row['data_collection_technology'] if pd.notna(
            row['data_collection_technology']) else ['None']
        metrics = row['metrics_recorded'] if pd.notna(
            row['metrics_recorded']) else ['None']
        sources = row['data_source'] if pd.notna(
            row['data_source']) else ['None']
        classLabel = row['high_level_labels']

        # Split data collection technologies and process each one
        tech_list = str(technologies).strip('[]').split(',')
        metric_list = str(metrics).strip('[]').split(',')
        source_list = str(sources).strip('[]').split(',')
        classLabel_list = str(classLabel).split(',')

        for tech in tech_list:
            tech = tech.strip().strip("'")
            for metric in metric_list:
                metric = metric.strip().strip("'")
                for source in source_list:
                    source = source.strip().strip("'")
                    for label in classLabel_list:
                        label = label.strip().lower()
                        processed_data.append([tech, metric, source, label] + [row[column] for column in row.index if column not in [
                                              'data_collection_technology', 'metrics_recorded', 'data_source', 'high_level_labels']])

    # Construct DataFrame
    df_processed = pd.DataFrame(processed_data, columns=['dataColl_techniques', 'metrics_recorded', 'data_source', 'high_level_labels'] + [
                                column for column in data.columns if column not in ['data_collection_technology', 'metrics_recorded', 'data_source', 'high_level_labels']])

    # Replace using dictionaries for data_collection_technique and network_type, and source categorization
    df_processed.replace({"dataColl_techniques": data_collection_technique_dict,
                         "metrics_recorded": network_to_category, "data_source": public_dataset}, inplace=True)

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
        data, columns=['ml_task_description', 'dataColl_techniques','data_source', 'metrics_recorded', 'high_level_labels'])
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


def retrieving_task_types(row):
    """
    Retrieves task types based on the task columns in the DataFrame.
    Args:
        row (Series): A row of DataFrame.
    Returns:
        str: Comma-separated string of cancer types.
    """
    # Get columns with value 1 and replace names using the dictionary
    task_types = [taskdict.get(col, col)
                  for col in task_columns if row[col] == 1]
    # Join the column names with a comma if more than one type is found
    return ', '.join(task_types)


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
    data_frame['DataCollection_technique'] = data_frame['dataColl_techniques']
    data_frame['DatasetName'] = data_frame['data_source'].replace(
        'None', np.nan)
    data_frame['DataSize_all_copy'] = data_frame['DataSize_all']
    data_frame['DataSize_all_copy'] = data_frame.apply(
        lambda row: f"{int(row['DataSize_all_copy'])} ({row['image_type']})" if pd.notna(
            row['DataSize_all_copy']) and row['DataSize_all_copy'] != '' else row['DataSize_all_copy'],
        axis=1
    )
    data_frame.replace({"image_type": image_typesdict}, inplace=True)
    data_frame['network_type_copy'] = data_frame['network_type']
    data_frame['multiple_cohorts_copy'] = data_frame['multiple_cohorts']
    # Define bins for binning the 'DataSize_all' column
    data_frame['DataSize_all'] = pd.to_numeric(
        data_frame['DataSize_all'], errors='coerce')
    # Fill NaN values with a specific value (for example, 0) before cutting into categories
    data_frame['DataSize_all'].fillna(0, inplace=True)
    data_frame['DataSize_all'] = pd.cut(data_frame['DataSize_all'].astype(float), bins=[
                                        0, 500, 1000, 5000, float('inf')], labels=[1, 2, 3, 4], include_lowest=True)

    # Get cancer types from subspecialty columns
    data_frame['cancer_type'] = data_frame.apply(
        retrieving_cancer_types, axis=1)
    # Get task types from subspecialty columns
    data_frame['task_type'] = data_frame.apply(retrieving_task_types, axis=1)
    # Convert 'article_date' to datetime format
    if data_frame['article_date'].dtype != 'datetime64[ns]':
        data_frame['article_date'] = pd.to_datetime(
            data_frame['article_date'], errors='coerce')
    # Set index for the DataFrame
    data_frame.set_index(['Paper_ID', 'title', 'doi', 'article_date', 'network_type', 'performance_auc', 'performance_sensitivity (recall)', 'performance_specificity',
                          'cancer_type', 'task_type', 'DataCollection_technique', 'DataSize_all_copy', 'raw_data_availability', 'code_availability', 'methodological',
                          'performance_cindex', 'performance_precision (PPV)', 'performance_NPV', 'performance_F1', 'performance_accuracy', 'benchmarking',
                          'implementation_detail', 'external_val_set', 'multiple_cohorts_copy', 'data_links', 'code_links', 'classes', 'DatasetName'], inplace=True)

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

def retrieving_top_neighbors(input, search_option='exact_match', k=15):
    """
    Retrieves top neighbors for a given input using KNN.
    Args:
        input_data (str): JSON string of input data.
        k (int): Number of neighbors to retrieve.
    Returns:
        DataFrame: DataFrame containing the top neighbors.
    """
    # Convert input data from JSON to a pandas Series
    json_dict = json.loads(input)
    keys_view = json_dict.keys()
    column_names = list(keys_view)
    column_names.append('performance_mean')
    input_data = pd.Series(json_dict)

    # Preprocess data
    data_frame = preprocessing_data()
    data_frame = data_frame[column_names]
    data_frame = cleaning_specific_columns(data_frame, column_names)

    # Sequential filtering based on the input_data fields in the required order
    for prefix in ['subspec_', 'dataColl_', 'high_', 'ml_']:
        columns_with_prefix = [col for col in column_names if col.startswith(prefix)]
        if columns_with_prefix:
            for col in columns_with_prefix:
                # Check if the column exists in input_data and its value is 1
                if col in input_data.index and input_data[col] == 1:
                    # Filter data_frame to only include rows where this column's value is 1
                    data_frame = data_frame[data_frame[col] == 1]

                    # If the filtered data_frame becomes empty, stop further filtering
                    if data_frame.empty:
                        return data_frame  # Return empty result if no matches are found

    if search_option == 'exact_match':
        # Apply filters for remaining columns that do not have the specified prefixes
        remaining_columns = [col for col in column_names if not col.startswith(('subspec_', 'dataColl_', 'high_', 'ml_', 'performance_'))]

        if remaining_columns:
            # Check if the column exists in input_data and its value is valid
            for col in remaining_columns:
                # Apply the filter based on the remaining column values
                data_frame = data_frame[data_frame[col] == input_data[col]]

                # If the filtered data_frame becomes empty, stop further filtering
                if data_frame.empty:
                    return data_frame  # Return empty result if no matches are found

        return qualityIndexForData(data_frame)  # Return the final filtered DataFrame
    
    else: 
        if len(data_frame) < k:
        # If fewer records are available than k, adjust k to the number of available records
        k = len(data_frame)

        # Apply KNN only if there are records remaining
        if k > 0:
            knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean').fit(data_frame)

            # Convert input_data to numpy array if it's a pandas Series
            if isinstance(input_data, pd.Series):
                input_data = input_data.values.reshape(1, -1)

            # Find indices of top neighbors
            distances, indices = knn.kneighbors(input_data)

            # Retrieve row names and performance means of top neighbors
            top_neighbors = data_frame.iloc[indices[0]]
            return qualityIndexForData(top_neighbors)
        else:
            # If no records are available after filtering, return an empty DataFrame
            return pd.DataFrame()
