# Import Streamlit for the web UI.
import streamlit as st
# Import the official OpenAI helper.
from openai import OpenAI
# Import PDF reader and in‑memory byte stream.
import PyPDF2, io


# ── Page header ─────────────────────────────────────────────────────────────
st.title('💬 PDF‑aware Chatbot')                                    # Visible page title.
st.write(                                                           # Short instructions.
    'Ask anything about an uploaded PDF.  \n'
    'You need an OpenAI API key – get one '
    '[here](https://platform.openai.com/account/api-keys).'
)


# ── API key ─────────────────────────────────────────────────────────────────
openai_api_key = st.text_input('OpenAI API Key', type='password')   # Hidden key box.
if not openai_api_key:                                             # Stop if empty.
    st.info('Please add your OpenAI API key to continue.', icon='🗝️')
    st.stop()

client = OpenAI(api_key=openai_api_key)                             # Create client.


# ── PDF upload & caching ────────────────────────────────────────────────────
pdf_file = st.file_uploader('📄 Upload a PDF to chat about', type='pdf')  # Upload.

# Read the PDF once, store its text in session_state.
if pdf_file and 'doc_text' not in st.session_state:
    reader   = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))        # Parse bytes.
    pages    = [p.extract_text() or '' for p in reader.pages]       # Text per page.
    fulltext = '\n'.join(pages)                                     # Join pages.
    st.session_state.doc_text = fulltext                            # Cache.
    st.success(f'Loaded {len(pages)} page(s).')                     # Confirmation.

# Optional preview of the extracted text.
if 'doc_text' in st.session_state:
    with st.expander('Preview extracted text', expanded=False):
        st.write(st.session_state.doc_text[:2000] + '…')            # First 2 k chars.


# ── Chat history ────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:                              # First run.
    st.session_state.messages = []                                  # Init history.

# Render past conversation.
for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])


# ── User input ─────────────────────────────────────────────────────────────
prompt = st.chat_input('Ask a question about the PDF…')             # Chat box.
if not prompt:                                                      # No input yet.
    st.stop()

# Store and display the new user message.
st.session_state.messages.append({'role': 'user', 'content': prompt})
with st.chat_message('user'):
    st.markdown(prompt)


# ── Build context for the model ────────────────────────────────────────────
context = []                                                        # Fresh list.

# Add document excerpt as a system message (only if a PDF is loaded).
if 'doc_text' in st.session_state:
    doc_excerpt = st.session_state.doc_text[:12000]                 # Trim for cost.
    context.append({
        'role': 'system',
        'content': (
            'You are a helpful assistant that responds in rap lyrics \n'
            'Answer the user strictly using the information in this document:\n'
            + doc_excerpt
        )
    })

# Append the running chat history.
context.extend(
    {'role': m['role'], 'content': m['content']}
    for m in st.session_state.messages
)


# ── Model call with streaming ──────────────────────────────────────────────
with st.chat_message('assistant'):                                  # Display area.
    stream = client.chat.completions.create(                        # Request GPT‑3.5.
        model='gpt-3.5-turbo',
        messages=context,
        stream=True,
    )
    reply = st.write_stream(stream)                                 # Show tokens live.

# Save assistant reply.
st.session_state.messages.append({'role': 'assistant', 'content': reply})