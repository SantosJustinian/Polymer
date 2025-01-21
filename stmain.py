# streamlit_app.py
import streamlit as st
from PyPDF2 import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

api_key = st.secrets["OPENAI_API_KEY"]


# Step 1: Extract text from PDF
def extract_pdf_text(file):
    raw_text = ''
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content + "\n"
    return raw_text

# Step 2: Split text into chunks and filter for relevant keywords
def split_and_filter_text(text, max_length=5000, keywords=["sustainability", "net zero", "emissions", "renewable", "ambition"]):
    chunks = []
    current_chunk = ""
    
    for paragraph in text.split("\n"):
        # Only add paragraphs containing keywords
        if any(keyword in paragraph.lower() for keyword in keywords):
            if len(current_chunk) + len(paragraph) > max_length:
                # Finalize the chunk and start a new one if limit is exceeded
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"
    
    # Add the final chunk if it has relevant text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Step 3: Summarize text using OpenAI's API
def summarize_text(text):
    client = OpenAI(api_key= api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "identify opportunities instead of just climate risks + gives them a rating based on the 6 pillars they have now + people centric features"},
            {"role": "user", "content": text}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

# Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>golden.ai</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;color: gray;'>The Ultimate Climate Screen</h3>", unsafe_allow_html=True)

    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Begin") and uploaded_files:
        final_summary = ""
        for uploaded_file in uploaded_files:
            final_summary = ""
            with st.spinner(f"Extracting text from PDF {uploaded_file.name}..."):
                # Extract text from uploaded PDF
                pdf_text = extract_pdf_text(uploaded_file)

                # Split and filter text based on keywords
                chunks = split_and_filter_text(pdf_text)

            st.success("Text extracted and filtered successfully!")
            with st.spinner(f"Loading Summary for {uploaded_file.name}..."):
                for chunk in chunks:
                    summary = summarize_text(chunk)
                    final_summary += summary + "\n"
                st.write(f"### Consolidated Summary for {uploaded_file.name}")
                st.write(final_summary)

if __name__ == "__main__":
    main()
