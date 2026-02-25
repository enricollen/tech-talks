"""
mcp run mcp_server/mcp_server.py --transport=streamable-http
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Local MCP Server - Recipes Assistant")

@mcp.tool()
def list_recipes() -> list[str]:
    """list all available italian recipes"""
    recipes = [
        "Lasagna",
        "Pasta Carbonara",
        "Pizza Margherita",
        "Risotto alla Milanese",
        "Tiramisu"
    ]
    return recipes


@mcp.tool()
def get_recipe_instructions(recipe_name: str) -> str:
    """get instructions for preparing a specific italian recipe"""
    # load recipe instructions from json file
    recipes_file = Path(__file__).parent / "recipes.json"
    
    if not recipes_file.exists():
        return f"Recipe instructions not available at the moment. Please try again later."
    
    try:
        with open(recipes_file, "r", encoding="utf-8") as f:
            recipes_data = json.load(f)
        
        # case-insensitive search
        recipe_name_lower = recipe_name.lower()
        for recipe_key, recipe_info in recipes_data.items():
            if recipe_key.lower() == recipe_name_lower:
                instructions = recipe_info.get("instructions", "no instructions available")
                ingredients = recipe_info.get("ingredients", [])
                
                result = f"**{recipe_key}**\n\n"
                result += f"ingredients:\n"
                for ingredient in ingredients:
                    result += f"- {ingredient}\n"
                result += f"\ninstructions:\n{instructions}"
                return result
        
        return f"recipe '{recipe_name}' not found. available recipes: {', '.join(recipes_data.keys())}"
    
    except Exception as e:
        return f"error reading recipe file: {str(e)}"


# add a resource with usage instructions
@mcp.resource("guide://usage")
def get_usage_guide() -> str:
    """get instructions on how to use this mcp server"""
    guide = """
    # italian recipes mcp server - usage guide
    
    this mcp server provides tools to explore and cook italian recipes.
    
    ## available tools:
    
    1. **list_recipes()**
       - returns a list of all available italian recipes
       - no parameters needed
       - use this first to see what recipes are available
    
    2. **get_recipe_instructions(recipe_name: str)**
       - returns detailed cooking instructions and ingredients for a specific recipe
       - parameter: recipe_name (case-insensitive)
       - example: get_recipe_instructions("lasagna")
    
    ## how to use:
    
    1. call list_recipes() to see available recipes
    2. choose a recipe you want to cook
    3. call get_recipe_instructions("recipe name") to get the full recipe
    
    ## available recipes:
    - lasagna
    - pasta carbonara
    - pizza margherita
    - risotto alla milanese
    - tiramisu
    
    enjoy cooking! 🍝
    """
    return guide


# add a prompt for recipe suggestions
@mcp.prompt()
def suggest_recipe(occasion: str = "dinner", dietary_preference: str = "none") -> str:
    """generate a prompt to get italian recipe suggestions based on occasion and dietary preferences"""
    base_prompt = f"suggest an italian recipe from the available recipes (lasagna, pasta carbonara, pizza margherita, risotto alla milanese, tiramisu) that would be perfect for {occasion}"
    
    if dietary_preference and dietary_preference.lower() != "none":
        base_prompt += f" considering {dietary_preference} dietary preferences"
    
    base_prompt += ". explain why this recipe is a good choice and provide the cooking instructions."
    
    return base_prompt