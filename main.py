#!/usr/bin/env python3
import sys
import asyncio
import warnings
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any

# Comprehensive warning suppression - only show print statements
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress specific library warnings
import logging
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

# Ensure output is flushed immediately (for PyInstaller builds)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    # Fallback for older Python versions or when reconfigure isn't available
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_mcp_m2m import MCPClientCredentials
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load .env file - from bundle when frozen, otherwise current directory
def extract_bundled_files():
    """Extract bundled data.json to exe directory if needed."""
    if not getattr(sys, 'frozen', False):
        return  # Not running as exe, skip extraction
    
    exe_dir = Path(sys.executable).parent
    
    # Extract data.json if bundled (but not .env - it stays in the bundle)
    if hasattr(sys, '_MEIPASS'):  # PyInstaller temporary directory
        bundle_dir = Path(sys._MEIPASS)
        
        # Extract data.json
        bundled_data_json = bundle_dir / 'data.json'
        target_data_json = exe_dir / 'data.json'
        if bundled_data_json.exists() and not target_data_json.exists():
            shutil.copy2(bundled_data_json, target_data_json)


def get_data_path() -> str:
    """Get the path to data.json file."""
    # Extract bundled files first if running as exe
    extract_bundled_files()
    
    # Check if running as compiled executable (PyInstaller, etc.)
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        exe_dir = Path(sys.executable).parent
        return str(exe_dir / 'data.json')
    else:
        # Running as script
        script_dir = Path(__file__).parent
        return str(script_dir / 'data.json')


# Load .env file - from bundle when frozen, otherwise current directory
extract_bundled_files()  # Extract data.json first
if getattr(sys, 'frozen', False):
    # Running as compiled executable - load .env from bundle
    if hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
        env_path = bundle_dir / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        else:
            # Fallback to exe directory
            exe_dir = Path(sys.executable).parent
            env_path = exe_dir / '.env'
            if env_path.exists():
                load_dotenv(env_path)
else:
    # Running as script
    load_dotenv()


def load_credentials() -> Dict[str, str]:
    """Load credentials from data.json file."""
    data_path = get_data_path()

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"data.json not found at {data_path}. Please create it with your credentials.")

    with open(data_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    client_id = data.get('editable', {}).get(
        'locus_client_id', {}).get('value', '')
    client_secret = data.get('editable', {}).get(
        'locus_client_secret', {}).get('value', '')

    if not client_id or not client_secret:
        raise ValueError(
            'LOCUS_CLIENT_ID and LOCUS_CLIENT_SECRET must be set in data.json')

    return {
        'client_id': client_id,
        'client_secret': client_secret
    }


async def process_query(query: str, client_id: str, client_secret: str) -> str:
    """Process a query using the Locus MCP agent."""
    # Create MCP client with Client Credentials
    client = MCPClientCredentials(
        {
            'locus': {
                'url': 'https://mcp.paywithlocus.com/mcp',
                'transport': 'streamable_http',
                'auth': {
                    'client_id': client_id,
                    'client_secret': client_secret
                }
            }
        }
    )

    # Connect and load tools
    await client.initialize()
    tools = await client.get_tools()

    # Create LLM and agent
    llm = ChatAnthropic(
        model='claude-sonnet-4-20250514',
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
    )

    agent = create_react_agent(llm, tools)

    # Run the query
    result = await agent.ainvoke({
        'messages': [HumanMessage(content=query)]
    })

    # Get the last message content
    last_message = result['messages'][-1]
    if hasattr(last_message, 'content'):
        return last_message.content
    else:
        return str(last_message)


def run_query(query: str):
    """Run the Locus MCP agent with a query."""
    if not query:
        print("Error: Query is required")
        sys.stdout.flush()
        return None
    
    try:
        # Load credentials from data.json
        credentials = load_credentials()
        
        # Process the query
        response = asyncio.run(process_query(
            query,
            credentials['client_id'],
            credentials['client_secret']
        ))
        
        return response
    except Exception as e:
        print(f"Error in run_query: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get port from data.json or use default
def get_port() -> int:
    """Get port from data.json or return default."""
    try:
        data_path = get_data_path()
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)
                port = data.get('editable', {}).get('port', {}).get('value')
                if port:
                    return int(port)
    except Exception:
        pass
    return 8911  # Default port


PORT = get_port()


@app.route('/run', methods=['POST'])
def run_endpoint():
    """Endpoint to run the Locus MCP agent with a query."""
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        
        if not query:
            return jsonify({"status": "error", "message": "Query parameter is required"}), 400
        
        result = run_query(query)
        if result is None:
            return jsonify({"status": "error", "message": "Failed to process query"}), 500
        
        return jsonify({"status": "success", "result": result}), 200
    except Exception as e:
        print(f"Error in run_endpoint: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_endpoint():
    """Health check endpoint."""
    try:
        # Try to load credentials to verify configuration
        credentials = load_credentials()
        return jsonify({
            "status": "healthy",
            "configured": True,
            "port": PORT
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "configured": False,
            "error": str(e),
            "port": PORT
        }), 200  # Still return 200 but indicate unhealthy status


if __name__ == "__main__":
    # Start Flask server
    print("Starting Flask server...")
    print("Endpoints available:")
    print("  POST /run - Process query (requires 'query' parameter in JSON body)")
    print("  GET /health - Health check")
    print(f"  Server running on port {PORT}")
    sys.stdout.flush()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
