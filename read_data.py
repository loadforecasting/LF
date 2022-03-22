import pandas as pd
import os.path


def read_data(file_path, time_column, multiple_sheets=False):
    """
    reads an excel or csv file into a pandas dataframe
    multiple_sheets = True if the excel file has multiple sheets
    """
    data = None
    file_extension = file_path.split('.')[-1]
    if(file_extension not in ['csv', 'xlsx']):
        raise Exception('File extension not supported. Use either xlsx or csv')
    
    if(os.path.isfile(file_path)):
        if(file_extension == 'csv'):
            data = pd.read_csv(file_path)
        elif(file_extension == 'xlsx'):
            if(multiple_sheets):
                df = pd.ExcelFile(file_path)
                res = len(df.sheet_names)
                frames = []
                for i in range(res):
                    data_i = pd.read_excel(file_path, sheet_name=i)
                    
                    data_i = data_i[~data_i.index.duplicated(keep='first')]
                    data_i.dropna(inplace=True)
                    
                    frames.append(data_i)
                data = pd.concat(frames)
            else:
                data = pd.read_excel(file_path)
            
            if("Unnamed: 0"  in (data.columns)):
                    data.rename(columns={'Unnamed: 0':time_column}, inplace=True)
            elif("Datum" in (data.columns) and "Zeit" in (data.columns)):
                data[time_column]= data.Datum.astype(str)+' '+data.Zeit.astype(str)
            
            data.set_index(time_column, inplace=True)
            data.index = data.index.astype('datetime64[ns]')
        
        return data
    else:
        return 'File does not exist or not accessible'
    
    return

        
        
        