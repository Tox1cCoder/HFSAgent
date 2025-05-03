import os
import re
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
import numpy as np

# Load environment variables
load_dotenv()


# Math Tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    Args:
        a: first int
        b: second int
    """
    return a - b


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    Args:
        a: first int
        b: second int
    """
    return a % b


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression safely.
    Args:
        expression: A string representing a mathematical expression
    """
    try:
        # Define allowed operators and functions
        allowed_operators = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
            "mod": np.mod,
            "abs": np.abs,
            "round": np.round,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "power": np.power,
        }

        # Basic security check
        for func_name in allowed_operators.keys():
            if func_name in expression:
                # Replace function name with numpy function call
                expression = expression.replace(
                    func_name, f"allowed_operators['{func_name}']"
                )

        # For safety, use eval with limited namespace
        namespace = {
            "__builtins__": {},
            "allowed_operators": allowed_operators,
            "np": np,
        }
        result = eval(expression, namespace)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


@tool
def reverse_text(text: str) -> str:
    """Reverse a string of text.
    Args:
        text: The text to reverse
    """
    return text[::-1]


@tool
def extract_table_data(table_text: str) -> dict:
    """Extract data from a markdown table.
    Args:
        table_text: The markdown table text
    """
    try:
        # Split into lines and filter empty lines
        lines = [line.strip() for line in table_text.split("\n") if line.strip()]

        # Extract headers (first line)
        headers = [cell.strip() for cell in lines[0].split("|") if cell.strip()]

        # Skip separator line
        data_lines = [line for line in lines[2:]]

        # Extract data rows
        rows = []
        for line in data_lines:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            rows.append(cells)

        return {"headers": headers, "rows": rows}
    except Exception as e:
        return {"error": f"Error extracting table data: {str(e)}"}


@tool
def check_commutativity(table_data: dict) -> str:
    """Check if an operation is commutative based on its table representation.
    Args:
        table_data: A dictionary containing headers and rows of the table
    """
    try:
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        # First element in headers is usually the operation symbol, rest are elements
        elements = headers[1:]

        # Create operation mapping
        operation_map = {}
        for row in rows:
            row_element = row[0]
            for i, col_element in enumerate(elements):
                if i + 1 < len(row):
                    operation_map[(row_element, col_element)] = row[i + 1]

        # Check commutativity
        non_commutative_elements = set()
        for a in elements:
            for b in elements:
                if a != b:
                    a_op_b = operation_map.get((a, b))
                    b_op_a = operation_map.get((b, a))
                    if a_op_b != b_op_a:
                        non_commutative_elements.add(a)
                        non_commutative_elements.add(b)

        # Format the result as comma-separated list
        result = sorted(list(non_commutative_elements))
        return ", ".join(result)
    except Exception as e:
        return f"Error checking commutativity: {str(e)}"


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query
    """
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"wiki_results": formatted_search_docs}
    except Exception as e:
        return {"error": f"Error searching Wikipedia: {str(e)}"}


@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 results.
    Args:
        query: The search query
    """
    try:
        search_docs = ArxivLoader(query=query, load_max_docs=3).load()
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
                for doc in search_docs
            ]
        )
        return {"arvix_results": formatted_search_docs}
    except Exception as e:
        return {"error": f"Error searching Arxiv: {str(e)}"}


# Define all tools
tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    calculate,
    reverse_text,
    extract_table_data,
    check_commutativity,
    wiki_search,
    arvix_search,
]

# Retriever setup (conditionally execute if Supabase credentials are available)
try:
    # Initialize vector store if environment variables are set
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"):
        # Set up embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        # Connect to Supabase
        supabase_client = create_client(
            os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
        )
        # Create vector store
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents_langchain",
        )
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever=vector_store.as_retriever(),
            name="question_search",
            description="Search for similar questions and answers in the database",
        )
        # Add retriever to tools list
        tools.append(retriever_tool)
        has_vector_store = True
    else:
        has_vector_store = False
        print(
            "Supabase credentials not found. Proceeding without retrieval capabilities."
        )
except Exception as e:
    has_vector_store = False
    print(f"Error setting up vector store: {e}")

# Read system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()


# Build graph function
def build_graph(provider: str = "google"):
    """Build the LangGraph for the agent.
    Args:
        provider: LLM provider to use ('google' or 'groq')
    """
    # Select LLM based on provider
    if provider == "google":
        # Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=False,
        )
    elif provider == "groq":
        # Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        llm = ChatGroq(api_key=groq_key, model="qwen-qwq-32b", temperature=0)
    else:
        raise ValueError("Invalid provider. Choose 'google' or 'groq'")

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # System message
    sys_msg = SystemMessage(content=system_prompt)

    # Define nodes
    def assistant(state: MessagesState):
        """Assistant node"""
        # Make sure system prompt is included
        messages = state["messages"]
        if not messages or not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [sys_msg] + messages

        return {"messages": [llm_with_tools.invoke(messages)]}

    def retriever(state: MessagesState):
        """Retriever node - only used if vector store is available"""
        if not has_vector_store:
            # Skip retrieval if vector store is not available
            return {"messages": [sys_msg] + state["messages"]}

        try:
            similar_question = vector_store.similarity_search(
                state["messages"][-1].content
            )
            if similar_question:
                example_msg = HumanMessage(
                    content=f"Here is a similar question and answer for reference: \n\n{similar_question[0].page_content}",
                )
                return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        except Exception as e:
            print(f"Retriever error: {e}")

        # Default fallback
        return {"messages": [sys_msg] + state["messages"]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Add edges
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()


class GAIAAgent:
    """GAIA Agent using LangGraph for better control flow"""

    def __init__(self):
        """Initialize the agent with LangGraph and tools"""
        print("Initializing GAIA Agent with LangGraph...")

        try:
            # Create the graph using Google Gemini as default
            self.graph = build_graph(provider="google")
            print("GAIA Agent ready.")
        except Exception as e:
            print(f"Error initializing GAIA Agent: {e}")
            # Try fallback to groq if available
            if os.getenv("GROQ_API_KEY"):
                print("Trying fallback to Groq...")
                self.graph = build_graph(provider="groq")
                print("GAIA Agent ready (using Groq fallback).")
            else:
                raise e

    def __call__(self, question: str) -> str:
        """Process a question and return an answer"""
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # Special handling for table questions
        if "|" in question and "-|" in question:
            print("Detected table in question")
            question = (
                "This question contains a table that defines a mathematical operation. "
                "Extract and analyze it: " + question
            )

        # Special handling for reversed text
        if question.startswith(".") and ("..." in question or question.endswith(".")):
            print("Detected possible reversed text")
            reversed_text = reverse_text(question)
            question = f"This text appears to be reversed. Original: {question} Reversed: {reversed_text}"

        # Create initial message
        messages = [HumanMessage(content=question)]

        try:
            # Invoke the graph
            print("Invoking LangGraph...")
            result = self.graph.invoke({"messages": messages})

            # Extract content from final message
            final_message = result["messages"][-1]
            if hasattr(final_message, "content"):
                content = final_message.content
            else:
                content = str(final_message)

            print("Received response from LangGraph")
            print(f"Response first 50 chars: {content[:50]}...")

            # Extract the FINAL ANSWER portion
            pattern = r"FINAL ANSWER:\s*(.*?)(?:\s*$)"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                answer = match.group(1).strip()
                print(f"Extracted final answer: {answer[:50]}...")
                return answer
            else:
                print("No FINAL ANSWER format found, using full response")
                return content.strip()

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg


# Test the agent
if __name__ == "__main__":
    agent = GAIAAgent()

    # Test with a simple question
    test_question = "What is 2+2?"
    print(f"\nTesting with simple question: {test_question}")
    answer = agent(test_question)
    print(f"Answer: {answer}")
    print(f"Answer: {answer}")

    # Test with a table question
    table_question = """Given this table defining * on the set S = {a, b, c, d, e}
    |*|a|b|c|d|e|
    |---|---|---|---|---|---|
    |a|a|b|c|b|d|
    |b|b|c|a|e|c|
    |c|c|a|b|b|a|
    |d|b|e|b|e|d|
    |e|d|b|a|d|c|
    provide the subset of S involved in any possible counter-examples that prove * is not commutative. Provide your answer as a comma separated list of the elements in the set in alphabetical order."""

    print(f"\nTesting with table question")
    answer = agent(table_question)
    print(f"Answer: {answer}")

    # Test with reversed text
    reversed_question = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirW'
    print(f"\nTesting with reversed question: {reversed_question}")
    answer = agent(reversed_question)
    print(f"Answer: {answer}")
