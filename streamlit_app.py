import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

# Pydantic Models for Structured Output


class PropertyMatch(BaseModel):
    id: str = Field(..., description="Unique property ID, with column name unique_property_id")
    projectName: Optional[str] = Field(None, description="Name of the real estate project")
    location: Optional[str] = Field(None, description="Project location or address")
    price: Optional[str] = Field(None, description="Price or price range")
    area: Optional[str] = Field(None, description="Total or built-up area details")
    pincode: Optional[str] = Field(None, description="Project pincode")
    type: Optional[str] = Field(None, description="Property type, e.g. apartment, villa, plot, etc.")
    landmark: Optional[str] = Field(None, description="Nearby landmark if available")
    amenities: Optional[str] = Field(None, description="Mentioned amenities if available")

class RAGAnswer(BaseModel):
    matching_projects: List[PropertyMatch] = Field(default_factory=list)
    unmatched_points: List[str] = Field(default_factory=list)
    explanation: str = Field(..., description="Reasoning or context explanation")


# Page Configuration


st.set_page_config(
    page_title="Real Estate AI Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sidebar - Configuration
# ============================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key Input
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Model Selection
    model_name = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash-lite"],
        index=0
    )
    
    # Number of results
    k_results = st.slider(
        "Number of Retrieved Properties",
        min_value=3,
        max_value=10,
        value=5,
        help="How many similar properties to retrieve from the database"
    )
    
    # Temperature
    temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values = more focused, Higher values = more creative"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    ### 💡 Example Queries
    - "3BHK flats with lift in Yashvant Seth Jadhav Marg"
    - "List projects near Subhash Nagar with lift"
    - "Show apartments under 1 crore with parking"
    - "Properties with gym and swimming pool"
    """)

# ============================================
# Initialize Session State
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

st.title("🏢 Real Estate AI Assistant")
st.markdown("Ask me anything about available properties!")


@st.cache_resource
def load_faiss_index(api_key):
    """Load FAISS index with embeddings"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )
        db = FAISS.load_local(
            "faiss_realestate_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db, embeddings
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return None, None


# RAG Chain Function


def create_rag_chain(api_key, model_name, temperature):
    """Create the RAG chain with parser"""
    
    parser = PydanticOutputParser(pydantic_object=RAGAnswer)
    
    prompt = ChatPromptTemplate.from_template("""
You are a Real Estate Expert Assistant helping a user find matching properties.

You will be given retrieved property data (from embeddings) and a user query.

Your job:
1. Identify properties that match **all** conditions in the query.
2. Return your answer strictly as JSON according to the provided format instructions.
3. If some query conditions are not met, list them under `unmatched_points`.
4. Never assume data not present in the retrieved context.
5. If nothing matches, leave `matching_projects` empty and explain why.

---
Retrieved Property Data:
{context}

User Query:
{question}

{format_instructions}
""")
    
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature
    )
    
    rag_chain = prompt | model | parser
    
    return rag_chain, parser

# Query Processing Function


def process_query(query, db, api_key, model_name, temperature, k_results):
    """Process user query and return structured response"""
    
    try:
        # Retrieve similar documents
        results = db.similarity_search(query, k=k_results)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Property ID: {doc.metadata.get('unique_property_ID', 'Unknown')}\n{doc.page_content}"
            for doc in results
        ])
        
        # Create RAG chain
        rag_chain, parser = create_rag_chain(api_key, model_name, temperature)
        
        # Prepare input
        input_data = {
            "context": context,
            "question": query,
            "format_instructions": parser.get_format_instructions()
        }
        
        # Get response
        response = rag_chain.invoke(input_data)
        
        return response, results
    
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None, None


# Display Functions


def display_property_card(prop: PropertyMatch):
    """Display a single property as a card"""
    with st.container():
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"**🏢 {prop.projectName or 'N/A'}**")
            st.markdown(f"**ID:** `{prop.id}`")
            st.markdown(f"**Type:** {prop.type or 'N/A'}")
            st.markdown(f"**📍 Location:** {prop.location or 'N/A'}")
        
        with col2:
            st.markdown(f"**💰 Price:** {prop.price or 'N/A'}")
            st.markdown(f"**📐 Area:** {prop.area or 'N/A'}")
            st.markdown(f"**📮 Pincode:** {prop.pincode or 'N/A'}")
            if prop.landmark:
                st.markdown(f"**🗺️ Landmark:** {prop.landmark}")
        
        if prop.amenities:
            st.markdown(f"**✨ Amenities:** {prop.amenities}")
        
        st.divider()

def display_response(response: RAGAnswer, results):
    """Display the complete RAG response"""
    
    # Matching Projects
    st.markdown("### ✅ Matching Projects")
    
    if response.matching_projects:
        st.success(f"Found {len(response.matching_projects)} matching properties")
        
        for prop in response.matching_projects:
            display_property_card(prop)
    else:
        st.warning("No matching projects found.")
    
    # Unmatched Points
    if response.unmatched_points:
        st.markdown("### ⚠️ Unmatched Points")
        for point in response.unmatched_points:
            st.warning(f"• {point}")
    
    # Explanation
    st.markdown("### 💡 Explanation")
    st.info(response.explanation)
    
    # Referenced Properties (Expander)
    with st.expander("📚 View Retrieved Property Data"):
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Property {i}:**")
            st.text(doc.page_content[:] )
            st.divider()


# Check API Key and Load Index


if not api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to continue.")
    st.stop()

# Load FAISS index
if st.session_state.db is None:
    with st.spinner("🔄 Loading property database..."):
        db ,embeddings= load_faiss_index(api_key)
        if db is not None:
            st.session_state.db = db
            st.session_state.embeddings = embeddings
            
            st.success("✅ Database loaded successfully!")
        else:
            st.error("❌ Failed to load database. Please check your API key and FAISS index path.")
            st.stop()


# Chat Interface


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            display_response(message["response"], message["results"])

# Chat input
if query := st.chat_input("Ask about properties (e.g., '3BHK with lift near Subhash Nagar')"):
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process query and display response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching properties..."):
            response, results = process_query(
                query,
                st.session_state.db,
                api_key,
                model_name,
                temperature,
                k_results
            )
            
            if response:
                display_response(response, results)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "response": response,
                    "results": results
                })
