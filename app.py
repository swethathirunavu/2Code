import streamlit as st
import PyPDF2
import docx
import json
import sqlite3
import hashlib
import uuid
from datetime import datetime
from typing import List, Dict, Any
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re
import time
import random
import io
import traceback

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Sensai AI - Doc→Quiz Bot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI with hackathon configuration
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

# Initialize OpenAI client with hackathon settings
def initialize_openai_client(api_key):
    """Initialize OpenAI client with hackathon configuration"""
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://agent.dev.hyperverge.org"
    )

# Test API connection
def test_api_connection(api_key):
    """Test if the API key and connection work"""
    try:
        client = initialize_openai_client(api_key)
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, can you say 'API working'?"}],
            max_tokens=10
        )
        return True, response.choices[0].message.content
    except Exception as e:
        return False, str(e)

# Database setup for sensai features
def init_sensai_db():
    conn = sqlite3.connect('sensai_quiz.db')
    c = conn.cursor()
    
    # Quiz attempts table
    c.execute('''CREATE TABLE IF NOT EXISTS quiz_attempts
                 (id INTEGER PRIMARY KEY, 
                  session_id TEXT,
                  user_id TEXT,
                  file_hash TEXT,
                  filename TEXT,
                  difficulty TEXT,
                  score INTEGER,
                  total_questions INTEGER,
                  timestamp TEXT,
                  chunks_covered TEXT,
                  wrong_questions TEXT,
                  time_taken REAL)''')
    
    # Question items table (item bank)
    c.execute('''CREATE TABLE IF NOT EXISTS question_items
                 (id INTEGER PRIMARY KEY,
                  file_hash TEXT,
                  question_text TEXT,
                  option_a TEXT,
                  option_b TEXT,
                  option_c TEXT,
                  option_d TEXT,
                  correct_answer TEXT,
                  explanation TEXT,
                  source_chunk TEXT,
                  page_number INTEGER,
                  chunk_id INTEGER,
                  difficulty TEXT,
                  citation TEXT)''')
    
    # Chat sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions
                 (id INTEGER PRIMARY KEY,
                  session_id TEXT,
                  user_id TEXT,
                  current_question INTEGER,
                  score INTEGER,
                  questions_order TEXT,
                  wrong_questions TEXT,
                  started_at TEXT,
                  completed_at TEXT)''')
    
    conn.commit()
    conn.close()

init_sensai_db()

# SIMPLIFIED SENSAI GPT Prompt Templates (More robust)
SENSAI_PROMPTS = {
    "easy": """Create 2 simple multiple choice questions from this text. Focus on basic facts and definitions.

Text: {chunk}

Format EXACTLY like this:
QUESTION: What is the main topic discussed?
A) Option 1
B) Option 2  
C) Option 3
D) Option 4
CORRECT: A
EXPLANATION: Brief explanation why A is correct
CITATION: Page {page_num}
---
QUESTION: [Second question]
A) Option 1
B) Option 2
C) Option 3  
D) Option 4
CORRECT: B
EXPLANATION: Brief explanation why B is correct
CITATION: Page {page_num}
---""",

    "medium": """Create 2 multiple choice questions that test understanding of this text.

Text: {chunk}

Format EXACTLY like this:
QUESTION: What is the main concept explained?
A) Option 1
B) Option 2
C) Option 3
D) Option 4
CORRECT: A
EXPLANATION: Explanation with reasoning
CITATION: Page {page_num}
---
QUESTION: [Second question]
A) Option 1
B) Option 2
C) Option 3
D) Option 4
CORRECT: B
EXPLANATION: Explanation with reasoning
CITATION: Page {page_num}
---""",

    "hard": """Create 2 analytical multiple choice questions from this text.

Text: {chunk}

Format EXACTLY like this:
QUESTION: What can be concluded from this information?
A) Option 1
B) Option 2
C) Option 3
D) Option 4
CORRECT: A
EXPLANATION: Detailed analytical explanation
CITATION: Page {page_num}
---
QUESTION: [Second question]
A) Option 1
B) Option 2
C) Option 3
D) Option 4
CORRECT: B
EXPLANATION: Detailed analytical explanation
CITATION: Page {page_num}
---"""
}

class SensaiDocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file) -> tuple[str, int, List[Dict]]:
        """Extract text from PDF with enhanced debugging"""
        try:
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            page_count = len(pdf_reader.pages)
            page_content = []
            
            st.info(f"📄 PDF has {page_count} pages")
            
            # Limit to first 10 pages for faster processing
            max_pages = min(10, page_count)
            
            for page_num in range(max_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text and len(page_text.strip()) > 50:  # Only pages with substantial content
                        clean_text = page_text.replace('\n\n', '\n').strip()
                        text += f"\n--- Page {page_num + 1} ---\n{clean_text}\n"
                        
                        page_content.append({
                            "page_num": page_num + 1,
                            "text": clean_text,
                            "word_count": len(clean_text.split())
                        })
                        st.success(f"✅ Processed page {page_num + 1} ({len(clean_text)} chars)")
                    else:
                        st.warning(f"⚠️ Page {page_num + 1} has little/no text")
                        
                except Exception as page_error:
                    st.error(f"❌ Error on page {page_num + 1}: {str(page_error)}")
                    continue
            
            st.success(f"📖 Extracted {len(text)} characters from {len(page_content)} pages")
            return text, page_count, page_content
            
        except Exception as e:
            st.error(f"❌ PDF processing error: {str(e)}")
            st.error(f"Error details: {traceback.format_exc()}")
            return "", 0, []

    @staticmethod
    def extract_text_from_docx(file) -> tuple[str, int, List[Dict]]:
        """Extract text from DOCX with enhanced debugging"""
        try:
            doc = docx.Document(file)
            text = ""
            page_content = []
            current_page = 1
            chars_per_page = 2000
            current_chars = 0
            current_page_text = ""
            
            st.info(f"📄 DOCX has {len(doc.paragraphs)} paragraphs")
            
            for para in doc.paragraphs:
                para_text = para.text.strip()
                if para_text:  # Only add non-empty paragraphs
                    para_text += "\n"
                    text += para_text
                    current_page_text += para_text
                    current_chars += len(para_text)
                    
                    if current_chars > chars_per_page:
                        if current_page_text.strip():  # Only add if has content
                            page_content.append({
                                "page_num": current_page,
                                "text": current_page_text.strip(),
                                "word_count": len(current_page_text.split())
                            })
                            st.success(f"✅ Processed page {current_page} ({len(current_page_text)} chars)")
                        
                        current_page += 1
                        current_chars = 0
                        current_page_text = ""
                        
                        if current_page > 10:  # Limit to 10 pages
                            break
            
            # Add final page
            if current_page_text.strip():
                page_content.append({
                    "page_num": current_page,
                    "text": current_page_text.strip(),
                    "word_count": len(current_page_text.split())
                })
                st.success(f"✅ Processed final page {current_page}")
            
            st.success(f"📖 Extracted {len(text)} characters from {len(page_content)} pages")
            return text, len(page_content), page_content
            
        except Exception as e:
            st.error(f"❌ DOCX processing error: {str(e)}")
            st.error(f"Error details: {traceback.format_exc()}")
            return "", 0, []

    @staticmethod
    def chunk_text_with_citations(text: str, page_content: List[Dict]) -> List[Dict[str, Any]]:
        """Split text with debugging"""
        try:
            if not text or len(text.strip()) < 100:
                st.error("❌ Text too short for chunking")
                return []
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,  # Smaller chunks for better processing
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            chunks = splitter.split_text(text)
            st.success(f"📝 Created {len(chunks)} text chunks")
            
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                # Find corresponding page
                page_num = 1
                for page in page_content:
                    if any(word in chunk.lower() for word in page["text"].lower().split()[:10]):
                        page_num = page["page_num"]
                        break
                
                if len(chunk.strip()) > 100:  # Only meaningful chunks
                    chunk_objects.append({
                        "id": i,
                        "text": chunk.strip(),
                        "page_num": page_num,
                        "word_count": len(chunk.split()),
                        "questions_generated": 0
                    })
                    
            st.success(f"✅ Created {len(chunk_objects)} valid chunks")
            return chunk_objects
            
        except Exception as e:
            st.error(f"❌ Chunking error: {str(e)}")
            return []

class SensaiQuizGenerator:
    @staticmethod
    def generate_mcqs_with_citations(chunk: Dict, difficulty: str = "medium") -> List[Dict]:
        """Generate MCQs with enhanced error handling and debugging"""
        if not st.session_state.openai_api_key:
            st.error("🔑 No API key found")
            return []
        
        try:
            # Prepare the chunk text (limit size)
            chunk_text = chunk['text'][:2000] if len(chunk['text']) > 2000 else chunk['text']
            
            prompt = SENSAI_PROMPTS[difficulty].format(
                chunk=chunk_text,
                page_num=chunk['page_num']
            )
            
            st.info(f"🤖 Generating questions for chunk {chunk['id']} (Page {chunk['page_num']})")
            st.info(f"📝 Chunk preview: {chunk_text[:100]}...")
            
            # Initialize client
            client = initialize_openai_client(st.session_state.openai_api_key)
            
            # Make API call with timeout and retry
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a quiz generator. Create exactly 2 multiple choice questions in the specified format."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=800,
                temperature=0.3,
                timeout=30
            )
            
            questions_text = response.choices[0].message.content
            st.success(f"✅ Got API response ({len(questions_text)} chars)")
            
            # Debug: Show raw response
            with st.expander(f"🔍 Debug: Raw API Response for Chunk {chunk['id']}", expanded=False):
                st.code(questions_text)
            
            # Parse questions
            questions = SensaiQuizGenerator.parse_questions_with_citations(
                questions_text, chunk, difficulty
            )
            
            if questions:
                st.success(f"✅ Successfully parsed {len(questions)} questions")
                return questions
            else:
                st.error(f"❌ Failed to parse questions from chunk {chunk['id']}")
                return []
                
        except Exception as e:
            st.error(f"❌ Error generating questions for chunk {chunk['id']}: {str(e)}")
            st.error(f"Error details: {traceback.format_exc()}")
            return []
    
    @staticmethod
    def parse_questions_with_citations(questions_text: str, source_chunk: Dict, difficulty: str) -> List[Dict]:
        """Parse with enhanced debugging"""
        try:
            questions = []
            
            # Split by --- or try alternative splitting
            question_blocks = questions_text.split("---")
            
            if len(question_blocks) < 2:
                # Try alternative splitting methods
                question_blocks = re.split(r'\n\s*\n(?=QUESTION:)', questions_text)
            
            st.info(f"🔍 Found {len(question_blocks)} question blocks")
            
            for i, block in enumerate(question_blocks):
                if not block.strip():
                    continue
                
                st.info(f"🔍 Processing block {i+1}: {block[:100]}...")
                
                # Extract question components using regex
                question_data = {"difficulty": difficulty, "chunk_id": source_chunk["id"]}
                
                # More flexible parsing
                question_match = re.search(r'QUESTION:\s*(.+?)(?=\nA\))', block, re.DOTALL)
                if question_match:
                    question_data["question"] = question_match.group(1).strip()
                
                # Extract options
                for option in ['A', 'B', 'C', 'D']:
                    pattern = f'{option}\\)\\s*(.+?)(?=\\n[ABCD]\\)|\\nCORRECT:|$)'
                    match = re.search(pattern, block, re.DOTALL)
                    if match:
                        question_data[option] = match.group(1).strip()
                
                # Extract correct answer
                correct_match = re.search(r'CORRECT:\s*([ABCD])', block)
                if correct_match:
                    question_data["correct"] = correct_match.group(1).strip()
                
                # Extract explanation
                exp_match = re.search(r'EXPLANATION:\s*(.+?)(?=\nCITATION:|$)', block, re.DOTALL)
                if exp_match:
                    question_data["explanation"] = exp_match.group(1).strip()
                
                # Extract citation
                cit_match = re.search(r'CITATION:\s*(.+?)(?=\n|$)', block)
                if cit_match:
                    question_data["citation"] = cit_match.group(1).strip()
                else:
                    question_data["citation"] = f"Page {source_chunk['page_num']}"
                
                # Validate question has all required fields
                required_fields = ["question", "A", "B", "C", "D", "correct", "explanation"]
                missing_fields = [field for field in required_fields if field not in question_data or not question_data[field]]
                
                if not missing_fields:
                    question_data["source_chunk"] = source_chunk['text'][:200] + "..."
                    question_data["page_num"] = source_chunk["page_num"]
                    question_data["id"] = len(questions)
                    questions.append(question_data)
                    st.success(f"✅ Successfully parsed question {len(questions)}")
                else:
                    st.warning(f"⚠️ Question {i+1} missing fields: {missing_fields}")
            
            return questions
            
        except Exception as e:
            st.error(f"❌ Parsing error: {str(e)}")
            st.error(f"Error details: {traceback.format_exc()}")
            return []

# Simplified main function for debugging
def main():
    st.title("🧠 Sensai AI - Debug Mode")
    st.markdown("### 🔍 Enhanced Debugging for Quiz Generation")
    
    # Sidebar with debug options
    with st.sidebar:
        st.header("🔧 Debug Configuration")
        
        # API Key input with testing
        if not st.session_state.openai_api_key:
            api_key = st.text_input("🔑 Hackathon API Key", type="password")
            if api_key:
                st.session_state.openai_api_key = api_key
                # Test API immediately
                with st.spinner("Testing API connection..."):
                    success, message = test_api_connection(api_key)
                    if success:
                        st.success(f"✅ API works! Response: {message}")
                    else:
                        st.error(f"❌ API failed: {message}")
                st.rerun()
        else:
            st.success("✅ API Key loaded!")
            if st.button("🧪 Test API Connection"):
                success, message = test_api_connection(st.session_state.openai_api_key)
                if success:
                    st.success(f"✅ API works! Response: {message}")
                else:
                    st.error(f"❌ API failed: {message}")
        
        # Difficulty
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=0)
        
        # Debug options
        st.subheader("🔍 Debug Options")
        show_chunks = st.checkbox("Show Text Chunks", True)
        show_api_calls = st.checkbox("Show API Calls", True)
        max_chunks = st.slider("Max Chunks to Process", 1, 5, 2)
    
    # Main content
    if 'debug_state' not in st.session_state:
        st.session_state.debug_state = 'upload'
    
    if st.session_state.debug_state == 'upload':
        st.header("📤 Upload Document for Debug Analysis")
        
        uploaded_file = st.file_uploader("Choose file", type=['pdf', 'docx'])
        
        if uploaded_file is not None:
            st.success(f"📁 File uploaded: {uploaded_file.name}")
            
            if st.button("🔍 Process Document", type="primary"):
                if not st.session_state.openai_api_key:
                    st.error("🔑 Please add API key first!")
                    return
                
                # Step 1: Extract text
                st.header("📖 Step 1: Text Extraction")
                if uploaded_file.type == "application/pdf":
                    text, page_count, page_content = SensaiDocumentProcessor.extract_text_from_pdf(uploaded_file)
                else:
                    text, page_count, page_content = SensaiDocumentProcessor.extract_text_from_docx(uploaded_file)
                
                if not text:
                    st.error("❌ No text extracted. Check your document.")
                    return
                
                # Step 2: Create chunks
                st.header("✂️ Step 2: Text Chunking")
                chunks = SensaiDocumentProcessor.chunk_text_with_citations(text, page_content)
                
                if not chunks:
                    st.error("❌ No chunks created. Text might be too short.")
                    return
                
                if show_chunks:
                    st.subheader("📝 Text Chunks Preview")
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3
                        with st.expander(f"Chunk {i} (Page {chunk['page_num']}) - {chunk['word_count']} words"):
                            st.write(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                
                # Step 3: Generate questions
                st.header("🧠 Step 3: Question Generation")
                all_questions = []
                
                for i, chunk in enumerate(chunks[:max_chunks]):
                    st.subheader(f"Processing Chunk {i+1}")
                    
                    questions = SensaiQuizGenerator.generate_mcqs_with_citations(chunk, difficulty)
                    if questions:
                        all_questions.extend(questions)
                        st.success(f"✅ Generated {len(questions)} questions from chunk {i+1}")
                        
                        # Show generated questions
                        for j, q in enumerate(questions):
                            with st.expander(f"Question {j+1} from Chunk {i+1}", expanded=True):
                                st.write(f"**Q:** {q['question']}")
                                st.write(f"**A)** {q['A']}")
                                st.write(f"**B)** {q['B']}")
                                st.write(f"**C)** {q['C']}")
                                st.write(f"**D)** {q['D']}")
                                st.write(f"**Correct:** {q['correct']}")
                                st.write(f"**Explanation:** {q['explanation']}")
                                st.write(f"**Citation:** {q.get('citation', 'N/A')}")
                    else:
                        st.error(f"❌ No questions generated from chunk {i+1}")
                    
                    # Add delay to avoid rate limits
                    if i < len(chunks[:max_chunks]) - 1:
                        time.sleep(2)
                
                # Final results
                st.header("📊 Final Results")
                if all_questions:
                    st.success(f"🎉 Successfully generated {len(all_questions)} questions!")
                    st.session_state.debug_questions = all_questions
                    
                    # Show summary
                    st.subheader("📈 Generation Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Questions", len(all_questions))
                    with col2:
                        st.metric("Chunks Processed", len(chunks[:max_chunks]))
                    with col3:
                        success_rate = len(all_questions) / (len(chunks[:max_chunks]) * 2) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                else:
                    st.error("❌ No questions could be generated!")
                    st.info("💡 Try with:")
                    st.info("- A different document")
                    st.info("- Easier difficulty level") 
                    st.info("- Check if your API key works")
                    st.info("- Make sure document has readable text")

if __name__ == "__main__":
    main()
