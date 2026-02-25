# test MCP clients
# https://developers.llamaindex.ai/python/examples/tools/mcp/
# https://github.com/docker/mcp-gateway?tab=readme-ov-file#usage
# https://github.com/modelcontextprotocol/python-sdk

import asyncio
import os
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, AgentStream, ToolCallResult, AgentWorkflow
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCKER_GATEWAY_TOKEN = os.getenv("DOCKER_GATEWAY_TOKEN", "")  # copy from docker gateway output

async def docker_mcp_servers_gateway():
    http_client = BasicMCPClient(
        "http://localhost:8080/mcp",
        headers={
            "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
            "Accept": "application/json, text/event-stream"
        } if DOCKER_GATEWAY_TOKEN else {}
    )

    # list available tools
    tools = await http_client.list_tools()
    #print(tools)

    """# call a tool
    #result = await http_client.call_tool("calculate", {"x": 5, "y": 10})

    # list available resources
    #resources = await http_client.list_resources()
    #print(resources)

    # read a resource
    #content, mime_type = await http_client.read_resource("config://app")

    # list available prompts
    #prompts = await http_client.list_prompts()

    # get a prompt
    #prompt_result = await http_client.get_prompt("greet", {"name": "World"})

    tools = await aget_tools_from_mcp_url("http://host.docker.internal:8080/mcp")
    print(tools)"""

    client = BasicMCPClient(
        "http://localhost:8080/mcp",
        headers={
            "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
            "Accept": "application/json, text/event-stream"
        } if DOCKER_GATEWAY_TOKEN else {}
    )
    
    # print tools
    available_tools = await client.list_tools()
    #print("Available tools:", available_tools)
    
    """# print resources
    resources = await client.list_resources()
    print("\nresources:", resources)
    
    # print prompts
    prompts = await client.list_prompts()
    print("\nprompts:", prompts)"""
    
    tools = await aget_tools_from_mcp_url(
        "http://localhost:8080/mcp",
        client=client,
        allowed_tools=["get_transcript"], # check on docker which tools names are available for the enabled MCP servers
    )
    #print(tools)
    
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    agent = ReActAgent(tools=tools, llm=llm, verbose=True)
    
    response = await agent.run(user_msg="get me a transcript of the video at https://www.youtube.com/watch?v=Fhy_VFMlE9s?") # "fetch the latest news from the web page https://www.lastampa.it/")
    print(response)

async def local_mcp_server():
    client = BasicMCPClient("http://localhost:8000/mcp")
    
    # list available tools to see what's available on the local mcp server
    available_tools = await client.list_tools()
    print(f"Available tools from server (total: {len(available_tools.tools)}):")
    for tool in available_tools.tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Schema: {tool.inputSchema}")
        print()
    
    # get tools using aget_tools_from_mcp_url - this converts them to proper llamaindex functiontool objects
    try:
        tools = await aget_tools_from_mcp_url(
            "http://localhost:8000/mcp",
            client=client,
            #allowed_tools=["list_recipes", "get_recipe_instructions"],
        )
        print(f"\nLoaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description}")
    except Exception as e:
        print(f"Error loading tools: {e}")
        import traceback
        traceback.print_exc()
        tools = []
    
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    agent = ReActAgent(tools=tools, llm=llm, verbose=True)
    
    print("\nAgent run:")
    response = await agent.run(user_msg="Whats the recipe for pizza margherita?")
    print(f"\nAgent response: {response}")

async def hybrid_mcp_agent():
    """
    create an agent that uses both docker gateway mcp and local mcp servers
    local server: recipes assistant (list_recipes, get_recipe_instructions)
    """
    print("=== initializing hybrid mcp agent ===\n")
    
    # connect to docker gateway mcp
    print("connecting to docker gateway mcp...")
    docker_client = BasicMCPClient(
        "http://localhost:8080/mcp",
        headers={
            "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
            "Accept": "application/json, text/event-stream"
        } if DOCKER_GATEWAY_TOKEN else {}
    )
    
    # connect to remote fastmcp app
    print("connecting to remote fastmcp app...")
    remote_client = BasicMCPClient("https://unnecessary-crimson-wildebeest.fastmcp.app/mcp")
    
    # connect to local mcp
    print("connecting to local mcp...")
    local_client = BasicMCPClient("http://localhost:8000/mcp")
    
    # list tools from each source
    print("\n--- listing docker gateway tools ---")
    docker_available_tools = await docker_client.list_tools()
    print(f"found {len(docker_available_tools.tools)} tools from docker gateway:")
    for tool in docker_available_tools.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    print("\n--- listing local mcp tools ---")
    local_available_tools = await local_client.list_tools()
    print(f"found {len(local_available_tools.tools)} tools from local mcp:")
    for tool in local_available_tools.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    print("\n--- listing remote fastmcp tools ---")
    remote_available_tools = await remote_client.list_tools()
    print(f"found {len(remote_available_tools.tools)} tools from remote fastmcp:")
    for tool in remote_available_tools.tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # list resources from remote mcp
    print("\n--- listing remote fastmcp resources ---")
    try:
        remote_resources = await remote_client.list_resources()
        print(f"found {len(remote_resources.resources)} resources from remote fastmcp:")
        for resource in remote_resources.resources:
            print(f"  - {resource.uri}: {resource.name}")
            if hasattr(resource, 'description') and resource.description:
                print(f"    description: {resource.description}")
    except Exception as e:
        print(f"error listing resources: {e}")
    
    # list prompts from remote mcp
    print("\n--- listing remote fastmcp prompts ---")
    try:
        remote_prompts = await remote_client.list_prompts()
        print(f"found {len(remote_prompts.prompts)} prompts from remote fastmcp:")
        for prompt in remote_prompts.prompts:
            print(f"  - {prompt.name}: {prompt.description}")
            if hasattr(prompt, 'arguments') and prompt.arguments:
                print(f"    arguments: {prompt.arguments}")
    except Exception as e:
        print(f"error listing prompts: {e}")
    
    # get tools from docker gateway
    print("\n--- loading docker gateway tools ---")
    docker_tools = await aget_tools_from_mcp_url(
        "http://localhost:8080/mcp",
        client=docker_client,
        allowed_tools=["get_transcript"], # specify which docker tools to use
    )
    print(f"loaded {len(docker_tools)} tools from docker gateway")
    
    # get tools from local mcp
    print("\n--- loading local mcp tools ---")
    try:
        local_tools = await aget_tools_from_mcp_url(
            "http://localhost:8000/mcp",
            client=local_client,
            allowed_tools=["list_recipes", "get_recipe_instructions"], # specify which local tools to use
        )
        print(f"loaded {len(local_tools)} tools from local mcp")
    except Exception as e:
        print(f"error loading local tools: {e}")
        local_tools = []
    
    # get tools from remote fastmcp
    print("\n--- loading remote fastmcp tools ---")
    remote_tools = await aget_tools_from_mcp_url(
        "https://unnecessary-crimson-wildebeest.fastmcp.app/mcp",
        client=remote_client,
        #allowed_tools=["get_weather"], # specify which remote tools to use
    )
    print(f"loaded {len(remote_tools)} tools from remote fastmcp")
    
    # combine all tools
    all_tools = docker_tools + local_tools + remote_tools
    print(f"\n=== total tools available to agent: {len(all_tools)} ===")
    for tool in all_tools:
        print(f"  - {tool.metadata.name}: {tool.metadata.description}")
    
    # create agent with combined tools
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    agent = ReActAgent(tools=all_tools, llm=llm, verbose=True)
    
    # test the hybrid agent with streaming to show reasoning and tool calls
    print("\n=== testing hybrid agent ===\n")
    
    # test 1: get pizza margherita recipe from local mcp server hosted on local machine
    print(">>> query 1: " + "how to make pizza margherita?" + "\n")
    handler1 = agent.run(user_msg="how to make pizza margherita?")
    
    async for ev in handler1.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[tool call: {ev.tool_name} with args {ev.tool_kwargs}]")
    
    response1 = await handler1
    print("\n\nfinal answer1:", str(response1))
    
    # test 2: echo message from remote server hosted on fastmcp cloud
    print("\n\n>>> query 2: " + "you should echo the user's message" + "\n")
    handler2 = agent.run(user_msg="you should echo the user's message")
    
    async for ev in handler2.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[tool call: {ev.tool_name} with args {ev.tool_kwargs}]")
    
    response2 = await handler2
    print("\n\nfinal answer2:", str(response2))
    
    # test 3: get transcript of video from remote mcp server enabled on docker gateway
    print("\n\n>>> query 3: " + "get the transcript of the video at https://www.youtube.com/watch?v=Fhy_VFMlE9s" + "\n")
    handler3 = agent.run(user_msg="get the transcript of the video at https://www.youtube.com/watch?v=Fhy_VFMlE9s")
    
    async for ev in handler3.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[tool call: {ev.tool_name} with args {ev.tool_kwargs}]")
    
    response3 = await handler3
    print("\n\nfinal answer3:", str(response3))


async def agent_example_huggingface_space():
    """
    test gradio flux mcp server for image generation
    """
    print("=== testing gradio flux mcp server ===\n")
    
    # connect to gradio mcp
    print("connecting to gradio flux mcp server...")
    gradio_client = BasicMCPClient("https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/")
    
    # list available tools
    print("\n--- listing gradio flux tools ---")
    try:
        available_tools = await gradio_client.list_tools()
        print(f"found {len(available_tools.tools)} tools from gradio flux server:")
        for tool in available_tools.tools:
            print(f"  - {tool.name}: {tool.description}")
            print(f"    schema: {tool.inputSchema}")
            print()
    except Exception as e:
        print(f"error listing tools: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # get tools using aget_tools_from_mcp_url
    print("\n--- loading gradio flux tools ---")
    try:
        tools = await aget_tools_from_mcp_url(
            "https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/",
            client=gradio_client,
        )
        print(f"loaded {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description}")
    except Exception as e:
        print(f"error loading tools: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # create agent with flux tools
    llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    agent = ReActAgent(tools=tools, llm=llm, verbose=True)
    
    # test the agent with image generation request
    print("\n=== testing flux image generation agent ===\n")
    print(">>> query: generate an image of a sunset over mountains\n")
    
    handler = agent.run(user_msg="generate an image of a sunset over mountains")
    
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[tool call: {ev.tool_name} with args {ev.tool_kwargs}]")
    
    response = await handler
    print("\n\nfinal answer:", str(response))
   
   
async def multiagent_triage_example():
    """
    multiagent triage example with react agents using mcp tools
    triage agent routes requests to specialized agents:
    - recipeagent: handles recipe queries using local mcp
    - weatheragent: handles weather queries using fastmcp
    - videoagent: handles video transcription using docker gateway
    - imageagent: handles image generation using hugging face space
    """
    print("=== multiagent triage workflow ===\n")
    
    # connect to docker gateway mcp
    print("connecting to docker gateway mcp...")
    docker_client = BasicMCPClient(
        "http://localhost:8080/mcp",
        headers={
            "Authorization": f"Bearer {DOCKER_GATEWAY_TOKEN}",
            "Accept": "application/json, text/event-stream"
        } if DOCKER_GATEWAY_TOKEN else {}
    )
    
    # connect to remote fastmcp
    print("connecting to remote fastmcp app...")
    remote_client = BasicMCPClient("https://unnecessary-crimson-wildebeest.fastmcp.app/mcp")
    
    # connect to local mcp
    print("connecting to local mcp...")
    local_client = BasicMCPClient("http://localhost:8000/mcp")
    
    # connect to gradio mcp (flux image generation)
    print("connecting to gradio flux mcp server...")
    gradio_client = BasicMCPClient("https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/")
    
    # get tools from docker gateway (video transcription)
    print("\n--- loading docker gateway tools ---")
    docker_tools = await aget_tools_from_mcp_url(
        "http://localhost:8080/mcp",
        client=docker_client,
        allowed_tools=["get_transcript"],
    )
    print(f"loaded {len(docker_tools)} tools from docker gateway")
    
    # get tools from local mcp (recipes)
    print("\n--- loading local mcp tools ---")
    try:
        local_tools = await aget_tools_from_mcp_url(
            "http://localhost:8000/mcp",
            client=local_client,
            allowed_tools=["list_recipes", "get_recipe_instructions"],
        )
        print(f"loaded {len(local_tools)} tools from local mcp")
    except Exception as e:
        print(f"error loading local tools: {e}")
        local_tools = []
    
    # get tools from remote fastmcp (weather)
    print("\n--- loading remote fastmcp tools ---")
    remote_tools = await aget_tools_from_mcp_url(
        "https://unnecessary-crimson-wildebeest.fastmcp.app/mcp",
        client=remote_client,
        allowed_tools=["get_weather"],
    )
    print(f"loaded {len(remote_tools)} tools from remote fastmcp")
    
    # get tools from gradio mcp (image generation)
    print("\n--- loading gradio flux tools ---")
    try:
        gradio_tools = await aget_tools_from_mcp_url(
            "https://hysts-mcp-flux-1-schnell.hf.space/gradio_api/mcp/",
            client=gradio_client,
        )
        print(f"loaded {len(gradio_tools)} tools from gradio flux server:")
        for tool in gradio_tools:
            print(f"  - {tool.metadata.name}: {tool.metadata.description[:100]}...")
    except Exception as e:
        print(f"error loading gradio flux tools: {e}")
        print("⚠️ the gradio flux server might be unavailable or slow to respond")
        gradio_tools = []
    
    # initialize llm
    llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    
    # create specialized agents
    # recipe agent: handles recipe-related queries
    recipe_agent = ReActAgent(
        name="RecipeAgent",
        description="handles recipe queries and cooking instructions",
        tools=local_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a recipe expert. you can list recipes and provide detailed cooking instructions. "
            "use your tools to answer recipe-related questions."
        ),
    )
    
    # weather agent: handles weather queries
    weather_agent = ReActAgent(
        name="WeatherAgent",
        description="provides weather information for any location",
        tools=remote_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a weather assistant. use your weather tool to provide current weather information "
            "for any location the user asks about."
        ),
    )
    
    # video agent: handles video transcription
    video_agent = ReActAgent(
        name="VideoAgent",
        description="transcribes youtube videos given the url",
        tools=docker_tools,
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a video transcription specialist. you can get transcripts from youtube videos. "
            "use your tools to transcribe videos from urls."
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
            "use your tools to generate images based on user prompts."
        ),
    )
    
    # triage agent: routes to appropriate specialist
    triage_agent = ReActAgent(
        name="TriageAgent",
        description="routes user requests to the appropriate specialist agent",
        tools=[],  # no tools, just routes to other agents
        llm=llm,
        verbose=True,
        system_prompt=(
            "you are a triage agent. your only job is to route requests to specialist agents.\n"
            "you do not have any tools to answer questions yourself - you MUST hand off every request.\n\n"
            "analyze the user's request and immediately hand off to the appropriate specialist:\n"
            "- RecipeAgent: for recipe queries, cooking instructions, food preparation\n"
            "- WeatherAgent: for weather information, forecasts for any location\n"
            "- VideoAgent: for transcribing youtube videos from urls\n"
            "- ImageAgent: for generating images from text descriptions\n\n"
            "IMPORTANT: always use the handoff tool to transfer the request. never attempt to answer directly."
        ),
        can_handoff_to=["RecipeAgent", "WeatherAgent", "VideoAgent", "ImageAgent"],
    )
    
    # wire agents together in workflow
    agent_workflow = AgentWorkflow(
        agents=[triage_agent, recipe_agent, weather_agent, video_agent, image_agent],
        root_agent=triage_agent.name,
        initial_state={},
    )
    
    # run the workflow with multiple test queries
    print("\n=== running multiagent triage workflow ===\n")
    
    # test 1: recipe query
    print(">>> query 1: how to make tiramisu?\n")
    resp1 = await agent_workflow.run(
        user_msg="how to make tiramisu?"
    )
    print("\n--- response 1 ---")
    print(resp1)
    
    # test 2: weather query
    print("\n\n>>> query 2: what's the weather in rome?\n")
    resp2 = await agent_workflow.run(
        user_msg="what's the weather in rome?"
    )
    print("\n--- response 2 ---")
    print(resp2)
    
    # test 3: video transcription query
    print("\n\n>>> query 3: get transcript from https://www.youtube.com/watch?v=Fhy_VFMlE9s\n")
    resp3 = await agent_workflow.run(
        user_msg="get transcript from https://www.youtube.com/watch?v=Fhy_VFMlE9s"
    )
    print("\n--- response 3 ---")
    print(resp3)
    
    # test 4: image generation query
    print("\n\n>>> query 4: generate an image of a sunset over mountains\n")
    resp4 = await agent_workflow.run(
        user_msg="generate an image of a sunset over mountains"
    )
    print("\n--- response 4 ---")
    print(resp4)

 
if __name__ == "__main__":
    #asyncio.run(docker_mcp_servers_gateway())
    #asyncio.run(local_mcp_server())
    #asyncio.run(hybrid_mcp_agent())
    #asyncio.run(agent_example_huggingface_space())
    asyncio.run(multiagent_triage_example())
