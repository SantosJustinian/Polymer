# streamlit_app.py
import streamlit as st
from PyPDF2 import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline

api_key = st.secrets["OPENAI_API_KEY"]
climate_sentiment_analyzer = pipeline("text-classification", model="climatebert/distilroberta-base-climate-sentiment")

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
            {"role": "system", "content": "Summarize the client's sustainability ambitions."},
            {"role": "user", "content": text}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content

def analyze_climate_sentiment(text):
    result = climate_sentiment_analyzer(text)
    return result
# Streamlit app
def main():
    st.title("Sustainability Report Summarizer")

    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if st.button("Begin") and uploaded_files:
        final_summary = ""
        for uploaded_file in uploaded_files:
            with st.spinner(f"Extracting text from PDF {uploaded_file.name}..."):
                # Extract text from uploaded PDF
                pdf_text = extract_pdf_text(uploaded_file)
                chunks = split_and_filter_text(pdf_text)

            st.success("Text extracted and filtered successfully!")
            with st.spinner(f"Loading Summary for {uploaded_file.name}..."):
                for chunk in chunks:
                    summary = summarize_text(chunk)
                    final_summary += summary + "\n"

                sentiment_result = analyze_climate_sentiment(final_summary)
                score = [name["score"] for name in sentiment_result]
                score = score[0]
                sentiment = [name["label"] for name in sentiment_result]
                star_rating = score * 10
                st.write(f"### Consolidated Summary for {uploaded_file.name}")
                st.write(final_summary)
                st.write(f"Sentiment: {sentiment[0]}")
                st.write(f"Rating: {score:2f}")
                st.write("⭐" * int(star_rating) + "☆" * (10 - int(star_rating)))

if __name__ == "__main__":
    main()
