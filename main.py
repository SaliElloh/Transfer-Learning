import os
import pandas as pd
import tarfile 

# csv_dir = 'C:\Users\sally\Downloads\hdbd.tar\hdbd_data'
csv_dir = 'c:/Users/selloh/Downloads/hdbd_data'
extracted_tar_files = 'C:/Users/selloh/Desktop/hdbd_data_x'
def process_csv_files(csv_dir):
    dfs = []
    
    for file in os.listdir(csv_dir):
        file_path  = os.path.join(csv_dir, file)
    
        print(file_path)

        with tarfile.open(file_path, 'r') as tar:
            print(f'extracting current file_path {file_path}')
            tar.extractall(extracted_tar_files)
        # if file.endswith('.csv'):
        #     file_path = os.path.join(csv_dir, file)
        #     print(file_path)
        #     # Read the CSV file into a DataFrame
        #     df = pd.read_csv(file_path)
        #     # Append the DataFrame to the list
        #     dfs.append(df)
    
    # # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dfs, ignore_index=True)
    
    # return combined_df

# # # Process the CSV files
combined_data = process_csv_files(csv_dir)

# # Display the combined DataFrame
# print(combined_data.head())




# import pandas as pd

# # Load CSV data
# data = pd.read_csv("path_to_csv_file.csv")

# # Inspect the first few rows
# print(data.head())

# # Check for missing values
# print(data.isnull().sum())

# # EXPLORATORY DATA ANALYSIS:

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Summary statistics
# print(data.describe())


# # Visualize distributions of numerical variables
# sns.histplot(data['ECGtoHR'], bins=20, kde=True)
# plt.title('Distribution of ECGtoHR')
# plt.xlabel('ECGtoHR')
# plt.ylabel('Frequency')
# plt.show()

# # Explore relationships between variables
# sns.pairplot(data[['Throttle', 'Steering', 'Brake', 'RPM', 'Speed']])
# plt.show()

# # Plot categorical variables
# sns.countplot(data['weather'])
# plt.title('Weather Distribution')
# plt.show()



# # Explore contextual variables
# sns.countplot(data['weather'])
# plt.title('Weather Distribution')
# plt.show()

# sns.boxplot(x='weather', y='Speed', data=data)
# plt.title('Speed Distribution by Weather')
# plt.show()
        

