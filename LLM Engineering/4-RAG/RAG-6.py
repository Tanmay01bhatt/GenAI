#Upload files in Streamlit

import streamlit as st

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Perform file processing here
    st.write("File uploaded successfully!")


#  if we want to restrict this upload functionality to accept only a certain format of files?

uploaded_files = st.file_uploader(
    "Choose files", 
    type=["txt", "pdf"], 
    accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Perform file processing here
        st.write(f"File {uploaded_file.name} uploaded successfully!")


# user input / question

question = st.text_input("Enter your question",  value = "Enter your question here")
st.write("The question is:", question)



# provide a clear and organized way for users to submit their queries and uploaded files.

with st.form(key='qa_form', clear_on_submit=True):
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Please upload a file and enter your question.")


# full pipeline : q/a app

# Specify the filename of your local image
image_filename = 'Educative.png'

# Use st.image to display the image
st.image(image_filename, use_column_width=True)


# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

with st.form(key='qa_form', clear_on_submit=True, border = False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Please upload a file and enter your question.")


