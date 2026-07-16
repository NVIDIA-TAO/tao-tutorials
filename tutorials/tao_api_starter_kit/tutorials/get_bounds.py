def get_bounds_of_field(schema, field_path):
    """
    Extract minimum, maximum, and valid_options for a given field path from the schema.
    
    Args:
        schema (dict): The full schema response from the API
        field_path (list): Path to the field e.g., ["train", "optim", "lr"]
    
    Returns:
        dict: Contains minimum, maximum, valid_options, and other metadata
    """
    
    def traverse_schema(current_schema, path_remaining):
        if not path_remaining:
            return current_schema
            
        current_field = path_remaining[0]
        remaining_path = path_remaining[1:]
        
        # Navigate through nested structure
        if "properties" in current_schema and current_field in current_schema["properties"]:
            return traverse_schema(current_schema["properties"][current_field], remaining_path)
        else:
            return None
    
    # Find the field definition in the schema
    field_def = traverse_schema(schema, field_path)
    
    if not field_def:
        return {"error": f"Field path {' -> '.join(field_path)} not found in schema"}
    
    # Extract metadata
    metadata = {
        "minimum": "Not defined",
        "maximum": "Not defined", 
        "valid_options": "Not defined",
        "type": field_def.get("type"),
        "default": field_def.get("default"),
        "description": field_def.get("description", field_def.get("title")),
        "math_condition": field_def.get("math_cond")
    }
    
    # Extract minimum/maximum constraints
    if "minimum" in field_def:
        metadata["minimum"] = field_def["minimum"]
    
    if "maximum" in field_def:
        metadata["maximum"] = field_def["maximum"]
        
    # Extract valid options from enum
    if "enum" in field_def:
        metadata["valid_options"] = field_def["enum"]
    
    # Parse math_cond for additional constraints
    if "math_cond" in field_def:
        math_cond = field_def["math_cond"]
        # Handle common patterns like "> 0.0", ">= 1", etc.
        if math_cond.startswith("> "):
            try:
                min_val = float(math_cond[2:])
                metadata["minimum"] = min_val
                metadata["minimum_exclusive"] = True
            except ValueError:
                pass
        elif math_cond.startswith(">= "):
            try:
                min_val = float(math_cond[3:])
                metadata["minimum"] = min_val
                metadata["minimum_exclusive"] = False
            except ValueError:
                pass
    
    return metadata
