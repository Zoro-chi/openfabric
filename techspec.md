# Technical Specification: Creative AI Pipeline

## 1. Architecture Overview

### 1.1 System Components

- **Local LLM Service**: Hugging Face Transformers-based model hosted locally
- **Openfabric Apps Integration**: Text-to-Image and Image-to-3D model integration
- **Memory System**: Short-term and long-term memory storage
- **API Layer**: REST API for interfacing with the system
- **User Interface**: Gradio-based UI for interactive experience

### 1.2 Infrastructure

- **Containerization**: Docker with Docker Compose orchestration
- **Services**:
  - `llm-service`: Local LLM serving container
  - `api-service`: Main application logic and API endpoints
  - `memory-service`: Vector database for semantic storage
  - `ui-service`: Web interface
  - `db-service`: SQLite for persistent storage

### 1.3 Data Flow

```
User Prompt → API → Local LLM → Enhanced Prompt → Text-to-Image App →
Image → Memory Storage → Image-to-3D App → 3D Model → User
```

## 2. Technology Stack

### 2.1 Core Technologies

- **Backend**: Python 3.10+
- **LLM Framework**: Hugging Face Transformers
- **Local Model**: DeepSeek Coder or Llama 2/3 (7B parameter version)
- **Memory**:
  - Short-term: In-memory session storage
  - Long-term: SQLite + ChromaDB for vector embeddings
- **UI**: Gradio for interactive interface
- **Infrastructure**: Docker + Docker Compose
- **Openfabric SDK**: For app integration and execution

### 2.2 Docker Configuration

- Multi-container setup with Docker Compose
- GPU passthrough for LLM acceleration (if available)
- Volume mounts for persistent data storage
- Environment variable configuration

## 3. Component Specifications

### 3.1 Local LLM Service

- **Model Selection**: DeepSeek 7B or Llama 2 7B
- **Quantization**: GGUF format with 4-bit quantization for efficiency
- **Inference Engine**: llama.cpp or Hugging Face Text Generation Inference
- **Prompt Templates**: System prompts for creative expansion
- **Container Specs**:
  - 8GB+ RAM allocation
  - GPU acceleration when available
  - HTTP API endpoint for inference

### 3.2 Memory System

- **Short-term Memory**:

  - In-memory session cache with TTL
  - Context tracking for multi-turn interactions
  - Reference tracking for generated assets

- **Long-term Memory**:
  - SQLite database for metadata and references
  - ChromaDB for semantic search and similarity retrieval
  - Schema:
    - `CreationRecords`: Tracks all generations with timestamps
    - `Assets`: Links to generated images and 3D models
    - `Prompts`: Original and expanded prompts
    - `VectorEmbeddings`: Semantic representations for search

### 3.3 Openfabric Integration

- **Dynamic Schema Handling**:

  - Auto-fetch manifest and schema from Openfabric API
  - Dynamic request construction based on schema requirements
  - Error handling and retry mechanisms

- **App Chaining Logic**:
  - Pipeline management for sequential execution
  - Intermediate result handling
  - Asset management and temporary storage

### 3.4 User Interface

- **Gradio Components**:
  - Text input for prompts
  - Image display for generated visuals
  - 3D model viewer with interactive controls
  - History browser with search capability
  - Generation settings adjustment panel

## 4. Implementation Plan

### 4.1 Phase 1: Core Infrastructure

- Set up Docker and Docker Compose environment
- Implement LLM service with Hugging Face model
- Create basic API structure
- Establish SQLite database for persistent storage

### 4.2 Phase 2: Openfabric Integration

- Implement dynamic schema and manifest handling
- Create the text-to-image pipeline
- Add image-to-3D model pipeline
- Implement results handling and storage

### 4.3 Phase 3: Memory System

- Develop short-term memory mechanism
- Implement long-term storage with SQLite
- Add ChromaDB for vector embeddings and similarity search
- Create memory retrieval and context management

### 4.4 Phase 4: User Interface

- Build Gradio UI components
- Implement asset browsing and history functionality
- Add interactive 3D model viewer
- Create comprehensive logging for debugging

### 4.5 Phase 5: Refinement and Testing

- Optimize LLM prompt engineering for creative expansions
- Performance tuning and resource optimization
- Comprehensive testing with varied inputs
- Documentation and deployment guides

## 5. Docker Configuration

### 5.1 Docker Compose Structure

```yaml
services:
  llm-service:
    build:
      context: ./llm-service
      dockerfile: Dockerfile
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    ports:
      - "8001:8000"
    environment:
      - MODEL_PATH=/app/models/deepseek-or-llama
      - QUANTIZATION=4bit
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  api-service:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
    environment:
      - LLM_SERVICE_URL=http://llm-service:8000
      - MEMORY_SERVICE_URL=http://memory-service:8000
    depends_on:
      - llm-service
      - memory-service

  memory-service:
    build:
      context: ./memory-service
      dockerfile: Dockerfile
    volumes:
      - ./data:/app/data
    ports:
      - "8002:8000"
    environment:
      - DB_PATH=/app/data/memory.db
      - VECTOR_DB_PATH=/app/data/vectordb
    depends_on:
      - db-service

  db-service:
    image: keinos/sqlite3
    volumes:
      - ./data:/data

  ui-service:
    build:
      context: ./ui-service
      dockerfile: Dockerfile
    ports:
      - "7860:7860"
    environment:
      - API_URL=http://api-service:8000
    depends_on:
      - api-service
```

### 5.2 Resource Requirements

- **Minimum Hardware**:

  - 16GB RAM
  - 4-core CPU
  - 20GB disk space
  - GPU recommended but not required

- **Optimized Hardware**:
  - 32GB RAM
  - 8-core CPU
  - NVIDIA GPU with 8GB+ VRAM
  - 50GB SSD storage

## 6. Development and Deployment Process

### 6.1 Development Workflow

1. Local development with Docker Compose
2. Unit testing individual components
3. Integration testing with mock Openfabric apps
4. End-to-end testing with real Openfabric integration

### 6.2 Deployment

1. Clone repository and navigate to project directory
2. Download required models to `./models` directory
3. Configure environment variables
4. Run `docker-compose up -d`
5. Access UI at http://localhost:7860

## 7. Integration Points

### 7.1 Local LLM Integration

- REST API for prompt enhancement
- Streaming response capability
- Request format:
  ```json
  {
    "prompt": "User input text",
    "system_message": "You are a creative assistant...",
    "max_tokens": 512
  }
  ```

### 7.2 Openfabric App Integration

- Dynamic manifest and schema fetching
- Request construction based on schema
- Response handling and asset management
- Error recovery and retry logic

### 7.3 Memory System Integration

- Store/retrieve session context
- Semantic search across historical generations
- Asset reference tracking
- API endpoints for history browsing

## 8. Bonus Features Implementation

### 8.1 FAISS/ChromaDB for Similarity Search

- Embedding generation using sentence-transformers
- Vector indexing and efficient similarity search
- Real-time query capability for "like the one I made before"

### 8.2 Local Browser for 3D Assets

- Three.js-based 3D viewer in Gradio
- Model format conversion if needed
- Download and sharing capabilities

### 8.3 Voice-to-Text Interaction

- WebSpeech API integration in UI
- Real-time transcription
- Voice command support

## 9. Testing and Quality Assurance

### 9.1 Test Cases

- Input validation
- Prompt enhancement quality
- Pipeline integrity
- Memory recall accuracy
- Edge case handling

### 9.2 Performance Metrics

- Response time benchmarks
- Memory usage monitoring
- Success rate tracking
- User satisfaction metrics

## 10. Expected Challenges and Mitigations

### 10.1 Potential Issues

- LLM resource constraints: Implement quantization and optimization
- Openfabric app failures: Add robust error handling and retries
- Complex prompt understanding: Refine system messages and prompt engineering
- Memory scaling: Implement efficient indexing and pruning

### 10.2 Mitigation Strategies

- Fallback mechanisms for each component
- Graceful degradation when resources are limited
- Progressive enhancement based on available capabilities
- Comprehensive logging for debugging

## 11. Timeline and Milestones

- **Week 1**: Core infrastructure and LLM integration
- **Week 2**: Openfabric integration and basic pipeline
- **Week 3**: Memory system implementation
- **Week 4**: UI development and refinement
- **Final Delivery**: Testing, documentation, and demonstration

---

This technical specification provides a comprehensive blueprint for implementing the AI Developer Challenge using Docker, Docker Compose, and HuggingFace for local LLM integration. The modular architecture ensures maintainability while the memory system enables persistent creative context across sessions.
