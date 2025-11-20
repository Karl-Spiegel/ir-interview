# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Metadata Discovery Agent** built with LangGraph - an AI-powered conversational agent for enterprise data discovery at Pepper Money (financial services). It reduces data discovery time from 2-3 hours to 5-10 minutes by intelligently searching across Snowflake, metadata catalogs, and business glossaries.

## Architecture

The project uses a **state-based workflow pattern** with LangGraph:

```
User Query → Intent Classification → Conditional Routing →
Parallel Searches (Snowflake/Catalog/Glossary) →
Result Compilation & Ranking → Permission Filtering →
Natural Language Response
```

### Core Components

1. **MetadataDiscoveryAgent** (`LangGraph/agent_orchestration.py`): Main orchestration class
2. **AgentState**: TypedDict managing conversation state with messages, intent, results, and user context
3. **Six workflow nodes**: classify_intent, search_snowflake, search_catalog, search_glossary, compile_results, generate_response
4. **Conditional routing**: Routes queries based on intent (technical → Snowflake, business → glossary, general → catalog)

### Key Technologies

- **LangGraph**: Agent orchestration and state management
- **Claude 3 Sonnet**: LLM via AWS Bedrock (`anthropic.claude-3-sonnet`)
- **Snowflake**: Data warehouse with INFORMATION_SCHEMA queries
- **Python 3**: Primary language

## Development Commands

### Installation

```bash
# Install required dependencies (no requirements.txt exists yet)
pip install langgraph langchain langchain_aws snowflake-connector-python
```

### Running the Agent

```python
from LangGraph.agent_orchestration import MetadataDiscoveryAgent
from langchain.schema import HumanMessage

agent = MetadataDiscoveryAgent()
result = agent.graph.invoke({
    "messages": [HumanMessage("Find data about loan defaults")],
    "user_context": {"access_level": "analyst", "department": "risk"}
})
```

### Environment Requirements

- AWS credentials configured for Bedrock access
- Snowflake connection credentials
- No .env file exists - credentials are currently expected to be set in environment

## Project Status

This is a **prototype/POC** demonstrating the architecture. Several helper methods are referenced but not fully implemented:
- `_extract_search_terms()`: Extract search terms from queries
- `_execute_snowflake_query()`: Execute Snowflake queries
- `_calculate_relevance_score()`: Score result relevance
- `_get_lineage()`: Retrieve data lineage
- `_apply_permission_filters()`: Apply access control
- `search_data_catalog()`: Search metadata catalog
- `search_business_glossary()`: Search business glossary

## File Structure

```
ir-interview/
├── LangGraph/
│   ├── agent_orchestration.py    # Main implementation (166 lines)
│   ├── README_LANGRAPH.md        # Detailed documentation
│   ├── architecture.html         # Interactive architecture diagram
│   └── architecture.ini          # ASCII architecture diagram
└── README.md                      # Minimal root readme
```

## Important Implementation Details

### State Management
The `AgentState` TypedDict manages:
- `messages`: Conversation history with `add_messages` annotation
- `query_intent`: Classification (technical/business/general)
- `data_sources`: Discovered datasets
- `metadata_results`: Search results from each source
- `user_context`: User metadata for permission filtering
- `final_response`: Generated natural language response

### Permission & Governance
- Results are filtered based on `user_context.access_level`
- Built-in audit trail capability for compliance
- Role-based access control (RBAC) enforced in `compile_results` node

### LangGraph Patterns
- Uses `StateGraph` with conditional edges based on intent
- Convergence pattern: All search nodes → compile_results
- Compiled graph for reproducibility and debugging

## Future Enhancements (from documentation)

- Implement RAG with vector embeddings for semantic search
- Add data quality metrics to responses
- Develop few-shot learning for domain-specific classification
- Complete implementation of all stub methods for production deployment