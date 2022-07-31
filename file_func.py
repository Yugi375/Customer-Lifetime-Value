import os
import shutil
import pandas as pd
import joblib

# function to create temporary csv data file from it's binary object
def create_file(file_name,data_bytes):
    
    with open("tmp_files/"+file_name, 'wb') as f:
        f.write(data_bytes)
        
  
#delete temporary files
def delete_tmp_files(folder_name):
    
	folder = folder_name
	for filename in os.listdir(folder):
		
		file_path = os.path.join(folder, filename)
		try:
			os.unlink(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
   
#function to perform transformations on the data
def transformations(df,rfm,bgf):
    final_dataset=df[['InvoiceDate','CustomerID','InvoiceNo','Quantity','UnitPrice','Country','StockCode','Description']]
    final_dataset["Total_Revenue"]=final_dataset["UnitPrice"]*final_dataset["Quantity"]
    
    final_dataset.drop("Description",axis=1,inplace=True)
    final_dataset.drop("StockCode",axis=1, inplace=True)
    final_data=rfm_model.transform(final_dataset[["CustomerID","InvoiceDate","InvoiceNo","Total_Revenue"]])
    final_data=pd.DataFrame(final_data,columns=["CustomerID","Recency","Frequency","Monetary"])
    final_dataset=pd.concat([final_dataset.drop([],axis=1),final_data],axis=1)
    X=final_dataset.iloc[:,1:]
    y=final_dataset.iloc[:,0]
    
    X_transformed=scaler.transform(X)
    X_transformed=pd.DataFrame(X_transformed,columns=X.columns)
    return X_transformed,y

#function to load the model
def load_models():
    
	rfm_model=joblib.load("Models/RFM.joblib")
	bgf=joblib.load("Models/bgf.joblib")
	return rfm_model,bgf