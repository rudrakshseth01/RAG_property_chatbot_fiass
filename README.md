# 🏢 Real Estate Property RAG System

An intelligent Real Estate AI Assistant powered by Retrieval-Augmented Generation (RAG) using Google's Gemini AI, LangChain, and FAISS vector database.

# its video demo link 

https://drive.google.com/file/d/1bPgdx2YVzpsqA1L_wZyj5MQ0FBBUTjYO/view?usp=sharing


## 📋 Overview

This project implements a conversational AI assistant that helps users find real estate properties based on natural language queries. It uses advanced embedding techniques and semantic search to match user requirements with available properties in the database.

## ✨ Features

- **🤖 AI-Powered Search**: Natural language property search using Google Gemini
- **🔍 Semantic Matching**: FAISS vector database for intelligent similarity search
- **📊 Structured Responses**: Pydantic-based output parsing for consistent results
- **💬 Chat Interface**: Interactive Streamlit-based conversational UI
- **⚙️ Customizable**: Adjustable model parameters (temperature, retrieval count)
- **📈 Batch Processing**: Efficient embedding generation with rate limiting

## 🛠️ Tech Stack

- **AI/ML**: Google Gemini (gemini-2.5-flash-lite, gemini-embedding-001)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Environment**: Python 3.12+

## 📁 Project Structure

```
realestate_property_rag/
│
├── streamlit_app.py              # Main Streamlit application
├── embedding.ipynb                # Jupyter notebook for creating embeddings
├── merging.ipynb                  # Data merging and preprocessing
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (API keys)
│
├── data/                         # Dataset directory
│   ├── project.csv
│   ├── ProjectAddress.csv
│   ├── ProjectConfiguration.csv
│   ├── ProjectConfigurationVariant.csv
│   └── *final_merged_realestate_data.csv
│
└── faiss_realestate_index/      # FAISS vector database
    └── index.faiss
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Google API Key (for Gemini AI)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd realestate_property_rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 📊 Data Preparation

1. **Prepare your data**
   - Place your CSV files in the `data/` directory
   - Run `merging.ipynb` to merge and preprocess the data

2. **Generate embeddings**
   - Open `embedding.ipynb`
   - Run all cells to create the FAISS vector index
   - The index will be saved in `faiss_realestate_index/`

   > **Note**: The embedding process uses batch processing with rate limiting to avoid API quota issues.

### 🎯 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The app should load automatically

## 💡 Usage

### Example Queries

Try these natural language queries:

- "3BHK flats with lift in Yashvant Seth Jadhav Marg"
- "List projects near Subhash Nagar with lift"
- "Show apartments under 1 crore with parking"
- "Properties with gym and swimming pool"
- "Find villas with 4 bedrooms and a garden"

### Configuration Options

The sidebar provides several configuration options:

- **Model Selection**: Choose the Gemini model
- **Number of Results**: Control how many properties to retrieve (3-10)
- **Temperature**: Adjust response creativity (0.0 - 1.0)
  - Lower values (0.0-0.3): More focused and deterministic
  - Higher values (0.7-1.0): More creative and diverse

## 🔧 Key Components

### 1. Embedding Generation (`embedding.ipynb`)

- Loads property data from CSV
- Creates embeddings using Google's `gemini-embedding-001`
- Implements batch processing with rate limiting
- Saves FAISS vector index

### 2. Streamlit App (`streamlit_app.py`)

**Main Features:**
- Chat interface for user queries
- Property card display with structured information
- Similarity search using FAISS
- Structured output using Pydantic models

**Data Models:**
```python
PropertyMatch:
  - id: Unique property ID
  - projectName: Project name
  - location: Address/location
  - price: Price range
  - area: Property area
  - type: Property type (apartment, villa, etc.)
  - amenities: Available amenities
```

### 3. RAG Pipeline

1. **User Query** → Streamlit input
2. **Embedding** → Convert query to vector
3. **Similarity Search** → FAISS retrieves top-k matches
4. **Context Building** → Prepare retrieved properties
5. **LLM Processing** → Gemini analyzes and structures response
6. **Display** → Show matching properties with explanations

## 🛡️ Error Handling

The application handles common errors:

- **API Quota Exceeded**: Batch processing with delays prevents rate limiting
- **Missing API Key**: Clear warning message
- **Database Load Failure**: Graceful error handling with user feedback
- **Parse Errors**: Fallback mechanisms for response parsing

## 📊 Data Schema

The system expects properties with the following attributes:

- `unique_property_ID`: Unique identifier
- `projectName`: Name of the project
- `location`: Property location/address
- `price`: Price information
- `area`: Area/size details
- `pincode`: Location pincode
- `type`: Property type
- `landmark`: Nearby landmarks
- `amenities`: Available facilities

## ⚙️ Configuration

### Rate Limiting (embedding.ipynb)

```python
batch_size = 50              # Documents per batch
delay_between_batches = 2    # Seconds between batches
```

Adjust these values based on your API quota:
- **Free Tier**: Use smaller batches (25-50) with longer delays (3-5s)
- **Paid Tier**: Increase batch size and reduce delays

### Model Parameters

```python
temperature = 0.2            # Response consistency
k_results = 5               # Number of retrieved properties
model_name = "gemini-2.5-flash-lite"
```

## 📈 Performance Tips

1. **Optimize Batch Size**: Balance between speed and API limits
2. **Adjust Temperature**: Lower for consistent results, higher for variety
3. **Fine-tune k_results**: More results = better context but slower processing
4. **Cache FAISS Index**: Use `@st.cache_resource` for faster reloads

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: 'langchain_core.pydantic_v1'**
```bash
pip install --upgrade langchain-core>=0.1.0
```

**2. ResourceExhausted (429 Error)**
- Reduce `batch_size` in `embedding.ipynb`
- Increase `delay_between_batches`
- Check your quota: [Google AI Usage](https://ai.dev/usage)

**3. FAISS Index Not Found**
- Run `embedding.ipynb` completely to generate the index
- Ensure `faiss_realestate_index/` directory exists

**4. Streamlit Connection Error**
- Check if port 8501 is available
- Try: `streamlit run streamlit_app.py --server.port 8502`

## 📝 Development

### Adding New Features

1. **New Property Attributes**: Update `PropertyMatch` model in `streamlit_app.py`
2. **Custom Prompts**: Modify the prompt template in `create_rag_chain()`
3. **Different Embeddings**: Change model in `GoogleGenerativeAIEmbeddings()`

### Testing

```bash
# Test embeddings
jupyter notebook embedding.ipynb

# Test app locally
streamlit run streamlit_app.py
```

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



**Made with ❤️ using Google Gemini & LangChain**
