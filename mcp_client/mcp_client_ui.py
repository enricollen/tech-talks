# gradio chat interface for mcp multiagent triage system
# pip install gradio

import asyncio
import os
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import aget_tools_from_mcp_url
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow, AgentStream, ToolCallResult
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCKER_GATEWAY_TOKEN = os.getenv("DOCKER_GATEWAY_TOKEN", "")  # copy from docker gateway output

# global variables to hold the workflow
agent_workflow = None

async def setup_agents():
    """initialize all mcp clients and agents"""
    global agent_workflow
    
    print("setting up mcp clients and agents...")
    
    # connect to docker gateway mcp (use localhost when running client outside docker)
    # note: copy the bearer token from docker gateway startup output and add to .env file
    if not DOCKER_GATEWAY_TOKEN:
        print("⚠️ warning: DOCKER_GATEWAY_TOKEN not set in .env file")
        print("   copy token from docker gateway output: 'Use Bearer token: Authorization: Bearer <token>'")
    
    docker_client = BasicMCPClient(
        "http://localhost:8080/mcp",
        headers={
            "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
            "Accept": "application/json, text/event-stream"
        } if DOCKER_GATEWAY_TOKEN else {}
    )
    
    # connect to remote fastmcp
    remote_client = BasicMCPClient("https://unnecessary-crimson-wildebeest.fastmcp.app/mcp")
    
    # connect to local mcp
    local_client = BasicMCPClient("http://localhost:8000/mcp")
    
    # connect to gradio mcp (flux image generation)
    gradio_client = BasicMCPClient("https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/")
    
    # get tools from docker gateway (video transcription)
    try:
        docker_tools = await aget_tools_from_mcp_url(
            "http://localhost:8080/mcp",
            client=docker_client,
            allowed_tools=["get_transcript"],
        )
        print(f"✅ loaded {len(docker_tools)} tools from docker gateway")
    except Exception as e:
        print(f"❌ error loading docker gateway tools: {e}")
        docker_tools = []
    
    # get tools from local mcp (recipes)
    try:
        local_tools = await aget_tools_from_mcp_url(
            "http://localhost:8000/mcp",
            client=local_client,
            allowed_tools=["list_recipes", "get_recipe_instructions"],
        )
        print(f"✅ loaded {len(local_tools)} tools from local mcp:")
        for tool in local_tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description}")
    except Exception as e:
        print(f"❌ error loading local tools: {e}")
        print("⚠️ make sure the local mcp server is running:")
        print("   mcp run mcp_server/mcp_server.py --transport=streamable-http")
        local_tools = []
    
    # get tools from remote fastmcp (weather)
    try:
        remote_tools = await aget_tools_from_mcp_url(
            "https://unnecessary-crimson-wildebeest.fastmcp.app/mcp",
            client=remote_client,
            allowed_tools=["get_weather"],
        )
        print(f"✅ loaded {len(remote_tools)} tools from remote fastmcp")
    except Exception as e:
        print(f"❌ error loading remote fastmcp tools: {e}")
        remote_tools = []
    
    # get tools from gradio mcp (image generation)
    try:
        gradio_tools = await aget_tools_from_mcp_url(
            "https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/",
            client=gradio_client,
        )
        print(f"✅ loaded {len(gradio_tools)} tools from gradio flux server:")
        for tool in gradio_tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description}")
    except Exception as e:
        print(f"❌ error loading gradio flux tools: {e}")
        print("⚠️ the gradio flux server might be unavailable or slow to respond")
        gradio_tools = []
    
    # initialize llm
    llm = OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    
    # check which agents we can create based on available tools
    print("\n=== agent creation status ===")
    print(f"recipe agent: {'✅ ready' if local_tools else '❌ no tools'}")
    print(f"weather agent: {'✅ ready' if remote_tools else '❌ no tools'}")
    print(f"video agent: {'✅ ready' if docker_tools else '❌ no tools'}")
    print(f"image agent: {'✅ ready' if gradio_tools else '❌ no tools'}")
    print()
    
    # create specialized agents
    recipe_agent = ReActAgent(
        name="RecipeAgent",
        description="handles recipe queries and cooking instructions",
        tools=local_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a recipe expert with access to italian recipes. "
            "you have two tools: 'list_recipes' to list all available recipes, and 'get_recipe_instructions' to get detailed instructions for a specific recipe. "
            "when a user asks for a recipe, use get_recipe_instructions with the recipe_name parameter. "
            "if conversation history is provided, use it to understand context and references from previous messages."
        ),
    )
    
    weather_agent = ReActAgent(
        name="WeatherAgent",
        description="provides weather information for any location",
        tools=remote_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a weather assistant. use your weather tool to provide current weather information "
            "for any location the user asks about. "
            "if conversation history is provided, use it to understand context and references from previous messages."
        ),
    )
    
    video_agent = ReActAgent(
        name="VideoAgent",
        description="transcribes youtube videos given the url",
        tools=docker_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a video transcription specialist. you can get transcripts from youtube videos. "
            "use your tools to transcribe videos from urls. "
            "if conversation history is provided, use it to understand context and references from previous messages."
        ),
    )
    
    # image agent: handles image generation using flux model
    image_agent = ReActAgent(
        name="ImageAgent",
        description="generates images from text prompts using flux model",
        tools=gradio_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are an image generation specialist using the flux model. you can create images from text descriptions. "
            "use your tools to generate images based on user prompts. "
            "if conversation history is provided, use it to understand context and references from previous messages."
        ),
    )
    
    # triage agent: routes to appropriate specialist
    triage_agent = ReActAgent(
        name="TriageAgent",
        description="routes user requests to the appropriate specialist agent",
        tools=[],
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a routing agent. you must immediately hand off EVERY request to a specialist agent.\n"
            "you have NO tools to answer questions - you can ONLY use handoff.\n\n"
            "routing rules:\n"
            "- recipes/cooking → hand off to RecipeAgent\n"
            "- weather queries → hand off to WeatherAgent\n"
            "- youtube videos → hand off to VideoAgent\n"
            "- image generation → hand off to ImageAgent\n\n"
            "CRITICAL: agent names (RecipeAgent, WeatherAgent, VideoAgent, ImageAgent) are NOT tools.\n"
            "you can ONLY transfer requests using the handoff tool, never call agent names directly.\n\n"
            "if conversation history is provided, use it for context."
        ),
        can_handoff_to=["RecipeAgent", "WeatherAgent", "VideoAgent", "ImageAgent"],
    )
    
    # wire agents together in workflow
    agent_workflow = AgentWorkflow(
        agents=[triage_agent, recipe_agent, weather_agent, video_agent, image_agent],
        root_agent=triage_agent.name,
        initial_state={},
    )
    
    print("agents setup complete!")
    return "✅ agents ready!"

async def chat(message, history):
    """async chat function that uses the multiagent workflow with memory and shows tool calls"""
    global agent_workflow
    
    if agent_workflow is None:
        yield "❌ agents not initialized. please restart the app."
        return
    
    try:
        # build context from last 3 messages in history
        context_messages = []
        if history:
            # get last 3 exchanges (up to 6 messages: 3 user + 3 assistant)
            recent_history = history[-6:] if len(history) > 6 else history
            
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                # handle multimodal content (in case of files or tuples)
                if isinstance(content, tuple):
                    content = f"[file: {content[0]}]"
                
                context_messages.append(f"{role}: {content}")
        
        # create enhanced message with context
        if context_messages:
            context_str = "\n".join(context_messages)
            enhanced_message = (
                f"[conversation history (last 3 exchanges):\n{context_str}]\n\n"
                f"current user request: {message}"
            )
        else:
            enhanced_message = message
        
        # run the workflow with streaming to show tool calls
        handler = agent_workflow.run(user_msg=enhanced_message)
        
        accumulated_response = ""
        
        async for ev in handler.stream_events():
            if isinstance(ev, AgentStream):
                # stream agent reasoning+response
                accumulated_response += ev.delta
                yield accumulated_response
            elif isinstance(ev, ToolCallResult):
                # show tool calls
                tool_info = f"\n\n🔧 **tool call:** `{ev.tool_name}`\n📝 **args:** `{ev.tool_kwargs}`\n\n"
                accumulated_response += tool_info
                yield accumulated_response
        
        # ensure we await the handler to complete properly
        await handler
        
        # yield final accumulated response (keeps all reasoning + tool calls)
        if accumulated_response:
            yield accumulated_response
        
    except Exception as e:
        yield f"❌ error: {str(e)}"


# create gradio interface
def create_interface():
    """create and launch gradio chat interface"""
    
    # setup agents on startup
    print("initializing mcp agents...")
    setup_message = asyncio.run(setup_agents())
    print(setup_message)
    
    # create chat interface
    demo = gr.ChatInterface(
        chat,
        type="messages",
        title="🤖 MCP Multiagent Triage Chatbot",
        description=(
            "chat with specialized mcp agents:\n\n"
            "🍕 **RecipeAgent** - get recipes and cooking instructions\n"
            "🌤️ **WeatherAgent** - check weather for any location\n"
            "🎥 **VideoTranscriptionAgent** - transcribe youtube videos\n"
            "🎨 **ImageAgent** - generate images using flux model\n\n"
            "the TriageAgent will automatically route your request to the right specialist!\n\n"
            "💾 **persistent history** - your conversations are saved locally in browser\n"
        ),
        examples=[
            "how to make tiramisu?",
            "what's the weather in rome?",
            "get transcript from https://www.youtube.com/watch?v=Fhy_VFMlE9s",
            "generate an image of a sunset over mountains",
        ],
        theme=gr.themes.Soft(),
        save_history=True,  # enable persistent chat history in browser
    )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()