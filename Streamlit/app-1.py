import streamlit as st
import pandas as pd
import os
from io import BytesIO

st.set_page_config(page_title="App 1",layout="wide")
st.title("CSV /Excel Converter")
st.write("Convert files from CSV to Excel and Vice Versa")

upload_files = st.file_uploader("Upload your file", type=['csv', 'xlsx'],accept_multiple_files=True)

if upload_files:
     for file in upload_files:
          file_ext = os.path.splitext(file.name)[-1].lower()

          # read the file
          if file_ext == '.csv':
               data = pd.read_csv(file)
          elif file_ext == '.xlsx':
               data = pd.read_excel(file)
          else:
               st.error("Unsupported file type:",{file_ext})
               continue
          
          st.write(f"File Name:{file.name}")

          # dispay the dataframe
          st.write("Preview the dataframe:")
          st.dataframe(data.head())

          # Data Cleaning Options
          st.subheader("Data Cleaning Options")
          if st.checkbox(f"Clean data for {file.name}"):
               col1,col2 = st.columns(2)
               with col1:
                    if st.button("Remove Duplicates"):
                         data = data.drop_duplicates()
                         st.write("Duplicates Removed")
                
               with col2:
                    if st.button("Fill Missing Numeric Values"):
                            numeric_cols = data.select_dtypes(include=['number']).columns
                            for col in numeric_cols:
                                data[col].fillna(data[col].mean(), inplace=True)
                            st.write("Missing Numeric Values Filled with Mean")


            # File Conversion Options
          st.subheader("File Conversion Options")
          conv_type = st.radio(f"Convert {file.name } to:",["CSV","Excel"],key=file.name)
          if st.button(f"Convert {file.name}"):
               buffer = BytesIO()
               if conv_type == 'CSV':
                    data.to_csv(buffer,index=False)
                    file_name = file.name.replace(file_ext,'.csv')
                    mime_type = 'text/csv'
                
               elif conv_type == 'Excel':
                    data.to_excel(buffer,index=False)
                    file_name = file.name.replace(file_ext,'.xlsx')
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
               buffer.seek(0)
               
               st.success("All files processed successfully!")
               #Download Button
               st.download_button(
                    label = f"Download {file_name}",
                    data = buffer,
                    file_name=file_name,
                    mime = mime_type
               )
          
               