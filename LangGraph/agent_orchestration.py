from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_aws import ChatBedrock
import snowflake.connector

class AgentState(TypedDict):
    """State container for metadata discovery agent"""
    messages: Annotated[List, add_messages]
    query_intent: str
    data_sources: List[dict]
    metadata_results: dict
    user_context: dict
    final_response: str

class MetadataDiscoveryAgent:
    def __init__(self):
        self.llm = ChatBedrock(model="anthropic.claude-3-sonnet")
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("search_snowflake", self.search_snowflake_metadata)
        workflow.add_node("search_catalog", self.search_data_catalog)
        workflow.add_node("search_glossary", self.search_business_glossary)
        workflow.add_node("compile_results", self.compile_results)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define edges with conditional routing
        workflow.set_entry_point("classify_intent")
        
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_search,
            {
                "technical": "search_snowflake",
                "business": "search_glossary",
                "general": "search_catalog"
            }
        )
        
        # Convergence pattern - all searches lead to compilation
        workflow.add_edge("search_snowflake", "compile_results")
        workflow.add_edge("search_catalog", "compile_results")
        workflow.add_edge("search_glossary", "compile_results")
        workflow.add_edge("compile_results", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user query intent using LLM"""
        user_query = state["messages"][-1].content
        
        classification_prompt = f"""
        Classify this data discovery query:
        Query: {user_query}
        
        Categories:
        - technical: Looking for specific tables/columns
        - business: Looking for business metrics/KPIs  
        - general: Exploring available data
        
        Return only the category.
        """
        
        intent = self.llm.invoke(classification_prompt).content.strip()
        state["query_intent"] = intent
        return state
    
    def search_snowflake_metadata(self, state: AgentState) -> AgentState:
        """Query Snowflake information schema for technical metadata"""
        query = """
        SELECT 
            table_catalog,
            table_schema,
            table_name,
            column_name,
            data_type,
            comment
        FROM information_schema.columns
        WHERE LOWER(column_name) LIKE %s 
           OR LOWER(comment) LIKE %s
        LIMIT 20
        """
        
        # Extract search terms from user query
        search_term = self._extract_search_terms(state["messages"][-1].content)
        
        # Execute Snowflake query (simplified for demo)
        results = self._execute_snowflake_query(query, search_term)
        
        state["metadata_results"]["snowflake"] = results
        return state
    
    def compile_results(self, state: AgentState) -> AgentState:
        """Compile and rank results from multiple sources"""
        all_results = state["metadata_results"]
        
        # Apply relevance scoring
        scored_results = []
        for source, results in all_results.items():
            for result in results:
                score = self._calculate_relevance_score(
                    result, 
                    state["messages"][-1].content,
                    state["user_context"]
                )
                scored_results.append({
                    "source": source,
                    "data": result,
                    "score": score,
                    "lineage": self._get_lineage(result)
                })
        
        # Sort by relevance and filter by permissions
        filtered_results = self._apply_permission_filters(
            scored_results,
            state["user_context"]["access_level"]
        )
        
        state["data_sources"] = sorted(
            filtered_results, 
            key=lambda x: x["score"], 
            reverse=True
        )[:10]
        
        return state
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Generate natural language response with discovered metadata"""
        if not state["data_sources"]:
            state["final_response"] = "No matching data sources found."
            return state
        
        response_prompt = f"""
        User asked: {state["messages"][-1].content}
        
        Found data sources: {state["data_sources"]}
        
        Generate a helpful response that:
        1. Lists the most relevant tables/datasets
        2. Explains what each contains
        3. Shows data lineage where relevant
        4. Suggests related datasets they might want
        
        Keep it concise and actionable.
        """
        
        response = self.llm.invoke(response_prompt).content
        state["final_response"] = response
        return state
    
    def route_search(self, state: AgentState) -> str:
        """Route to appropriate search based on intent"""
        return state["query_intent"]

# Usage
agent = MetadataDiscoveryAgent()
result = agent.graph.invoke({
    "messages": [HumanMessage("Find data about loan defaults in Q3")],
    "user_context": {"access_level": "analyst", "department": "risk"}
})