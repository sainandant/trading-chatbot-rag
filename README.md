# 🤖 Trading Assistant Chatbot with RAG

An intelligent chatbot that answers trading and finance questions using Retrieval-Augmented Generation (RAG) architecture.

## 🌟 Features

- **Vector Search**: Uses ChromaDB for semantic document retrieval
- **AI-Powered Answers**: Leverages Hugging Face's FLAN-T5 model
- **Context-Aware**: Only answers based on provided knowledge base
- **Source Citations**: Shows which documents were used to generate answers
- **Interactive UI**: Built with Streamlit for easy interaction

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **ChromaDB** - Vector database
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **Hugging Face Transformers** - LLM (FLAN-T5-base)

## 📦 Installation
```bash
# Clone or download this project
cd trading-chatbot-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install streamlit chromadb sentence-transformers transformers pypdf torch
```

## 🚀 Usage
```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 How It Works

1. **Document Loading**: Reads trading knowledge from `documents/trading_basics.txt`
2. **Embedding Creation**: Converts text into vector embeddings using Sentence Transformers
3. **Vector Storage**: Stores embeddings in ChromaDB for fast semantic search
4. **Query Processing**: When user asks a question:
   - Converts question to embedding
   - Searches for most relevant documents
   - Sends context + question to FLAN-T5
   - Returns AI-generated answer with sources

## 📊 Architecture
```
User Question → Embedding Model → Vector Search (ChromaDB) 
→ Retrieve Relevant Docs → LLM (FLAN-T5) → Generated Answer
```

## 🎯 Use Cases

- Customer support automation
- Trading education platform
- Internal knowledge base assistant
- Onboarding new traders
- 24/7 FAQ handling

## 🔮 Future Enhancements

- [ ] Add PDF document upload capability
- [ ] Support for multiple document sources
- [ ] Conversation memory for follow-up questions
- [ ] Fine-tune model on Deriv-specific trading data
- [ ] Multi-language support
- [ ] Deploy to cloud (Streamlit Cloud/Hugging Face Spaces)

## 📝 Sample Questions

- "What is CFD trading?"
- "Explain leverage in trading"
- "What is risk management?"
- "What is a pip in forex?"

## 🤝 Contributing

Feel free to fork and improve! This is a learning project demonstrating RAG architecture.

## 📄 License

MIT License - Free to use and modify

---

**Built as part of AI Engineer application portfolio for Deriv**
