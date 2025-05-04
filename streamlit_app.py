import streamlit as st
from openai import OpenAI
import PyPDF2, io

# ── Page header ────────────────────────────────────────────────────────────────
st.title('💬 PDF‑aware Chatbot')
st.write(
    'Ask anything about an uploaded PDF.  \n'
    'You need an OpenAI API key – get one '
    '[here](https://platform.openai.com/account/api-keys).'
)

# ── API key ────────────────────────────────────────────────────────────────────
openai_api_key = st.text_input('OpenAI API Key', type='password')
if not openai_api_key:
    st.info('Please add your OpenAI API key to continue.', icon='🗝️')
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ── PDF upload & caching ───────────────────────────────────────────────────────
pdf_file = st.file_uploader('📄 Upload a PDF to chat about', type='pdf')

if pdf_file and 'doc_text' not in st.session_state:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    pages   = [p.extract_text() or '' for p in reader.pages]
    fulltext = '\n'.join(pages)
    st.session_state.doc_text = fulltext
    st.success(f'Loaded {len(pages)} page(s).')

if 'doc_text' in st.session_state:
    with st.expander('Preview extracted text', expanded=False):
        st.write(st.session_state.doc_text[:2000] + '…')

# ── Chat history ───────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])

# ── User input ────────────────────────────────────────────────────────────────
prompt = st.chat_input('Ask a question about the PDF…')
if not prompt:
    st.stop()

st.session_state.messages.append({'role': 'user', 'content': prompt})
with st.chat_message('user'):
    st.markdown(prompt)

# ── Build context for the model ───────────────────────────────────────────────
context = []
if 'doc_text' in st.session_state:
    doc_excerpt = st.session_state.doc_text[:12000]          # keep token usage reasonable
    context.append({
        'role': 'system',
        'content': (
            'You are a helpful assistant that responds in rap lyrics  '
            'Answer the user strictly using the information in this document:\n' +
            doc_excerpt
        )
    })

context.extend(
    {'role': m['role'], 'content': m['content']}
    for m in st.session_state.messages
)

# ── Model call with streaming ─────────────────────────────────────────────────
with st.chat_message('assistant'):
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=context,
        stream=True,
    )
    reply = st.write_stream(stream)

st.session_state.messages.append({'role': 'assistant', 'content': reply})