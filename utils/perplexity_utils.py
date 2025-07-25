from dotenv import load_dotenv
import requests
import re


def get_project_basic_info_perplexity(project_name: str, api_key: str, model: str = "sonar-pro"):
    """
    Query Perplexity API for basic real estate project info using the project name.
    
    Args:
        project_name (str): Name of the real estate project.
        api_key (str): Your Perplexity API key.
        model (str): The model to use. Default is "sonar-pro".
    
    Returns:
        dict: Parsed JSON response from Perplexity API with the info or error details.
    """
    if not api_key:
        raise ValueError("API key must be provided.")
    if not project_name:
        raise ValueError("Project name must be provided.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Enhanced system prompt for Vietnamese real estate analysis
    system_prompt = (
        "You are a Vietnamese real estate market expert with extensive knowledge of property development projects, "
        "pricing structures, and market conditions across Vietnam's major cities. You have access to current market data "
        "and can make intelligent estimates based on location, project type, and comparable developments in the area."
    )
    
    # Comprehensive user message with detailed instructions
    user_message = f"""Please analyze the Vietnamese real estate project named '{project_name}' and provide comprehensive information.

**SEARCH STRATEGY:**
1. First, search for direct information about '{project_name}'
2. If exact data is not available, analyze similar projects in the same area/district
3. Use comparable projects from the same developer if available
4. Apply Vietnamese real estate market standards and regional pricing patterns

**REQUIRED INFORMATION TO EXTRACT/ESTIMATE:**

**1. PROJECT BASIC INFO:**
- Full project name and alternative names
- Developer/owner company
- Exact location (district, city, address if available)
- Project type (apartment, villa, mixed-use, etc.)
- Current status (planning, under construction, completed, selling)
- Launch year and completion timeline

**2. PROJECT SCALE & SPECIFICATIONS:**
- Total number of units (apartments, villas, townhouses)
- Average unit size in m² (break down by unit type if mixed)
- Net Sellable Area (NSA) in m² total
- Gross Floor Area (GFA) in m² total
- Land area in m² (site area)
- Number of buildings/blocks/phases

**3. PRICING INFORMATION:**
- Current average selling price per m² (VND/m²)
- Price range if available (min-max per m²)
- Recent pricing trends or changes
- Price per unit (if available, specify unit type and size)

**4. CONSTRUCTION & DEVELOPMENT COSTS:**
- Estimated construction cost per m² (based on project type and location)
- Land cost per m² (based on area land values)
- Development timeline and phases

**ESTIMATION GUIDELINES WHEN EXACT DATA IS NOT AVAILABLE:**

**For TOTAL UNITS:** 
- High-rise apartments: 20-40 units per floor, 20-50 floors typical
- Mid-rise apartments: 4-8 units per floor, 5-15 floors typical  
- Villa/townhouse projects: Based on land area ÷ typical plot size (150-300m² per unit)
- Mixed-use: Estimate based on GFA and typical unit sizes

**For AVERAGE UNIT SIZE:**
- Ho Chi Minh City apartments: 60-120m² (luxury: 80-150m²)
- Hanoi apartments: 65-110m² (luxury: 90-140m²)
- Secondary cities: 70-130m² (more spacious)
- Villas/townhouses: 150-400m² (premium: 200-500m²)

**For SELLING PRICE PER M²:**
- Research recent transactions in the same district/area
- Consider project positioning (affordable, mid-range, luxury, ultra-luxury)
- Account for location premiums (central vs suburban)
- Use comparable projects' pricing as baseline

**For GROSS FLOOR AREA (GFA):**
- Calculate: Total units × Average unit size × Efficiency factor (1.3-1.5 for apartments, 1.1-1.3 for villas)
- Include common areas, corridors, amenities, parking

**For LAND AREA:**
- Urban apartments: GFA/Land ratio typically 3-8 (higher in central areas)
- Suburban/villa projects: GFA/Land ratio typically 0.3-1.5
- Check local zoning regulations and typical plot ratios

**For CONSTRUCTION COST PER M²:**
- Basic apartments: 15-25 million VND/m²
- Mid-range apartments: 20-35 million VND/m²  
- Luxury apartments: 30-50 million VND/m²
- Ultra-luxury/premium: 45-80+ million VND/m²
- Villas: 25-60 million VND/m² (depending on finishes)

**For LAND COST PER M²:**
- Research recent land auction prices in the area
- Use government published land price frameworks
- Consider location premiums and development rights

**RESPONSE FORMAT (PROVIDE EXACT NUMBERS ONLY):**

Info: [Detailed project description including developer, location, type, status, and any relevant background information]

Total Units: [NUMBER ONLY - no commas or text]
Average Unit Size: [NUMBER ONLY - in m²] 
Average Selling Price: [NUMBER ONLY - VND per m²]
Gross Floor Area: [NUMBER ONLY - total m²]
Construction Cost per sqm: [NUMBER ONLY - VND per m² for construction]
Land Area: [NUMBER ONLY - total land area in m²]
Land Cost per sqm: [NUMBER ONLY - VND per m² for land]

Sources: [List your sources - web results, comparable projects, or "Market analysis based on area comps"]
Confidence: [High/Medium/Low - based on data availability]

Analysis Method: [Explain how you derived each number - "Found exact data" OR "Estimated based on [comparable projects/area standards/project type]"]

Unit Size Analysis: [Explain your unit size calculation: mix of unit types, size distribution, etc.]

Pricing Analysis: [Explain your pricing calculation: recent comps, location factors, premium/discount factors]

Construction Cost Analysis: [Explain cost estimates: project type, quality level, location factors]

Land Cost Analysis: [Explain land cost estimates: area benchmarks, zoning, development intensity]

**IMPORTANT REQUIREMENTS:**
- Always provide numerical estimates even if exact data is not available
- Clearly distinguish between confirmed data and estimates
- Use 2024-2025 Vietnamese market conditions
- Consider inflation and recent market trends
- Be specific about your estimation methodology
- For mixed-use projects, provide weighted averages
- Account for project phasing if applicable

**LOCATION-SPECIFIC CONSIDERATIONS:**
- Ho Chi Minh City: Higher density, premium pricing in central districts
- Hanoi: Government influence, established vs new urban areas  
- Da Nang: Resort/tourism factors, beachfront premiums
- Secondary cities: Lower costs, larger units, emerging markets

Please provide all requested information with your best professional estimates where exact data is not available."""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 2000,  # Increased for comprehensive response
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False
    }

    # Debug: Log the request payload
    print(f"🔍 DEBUG: Sending request to Perplexity API")
    print(f"URL: {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {payload}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Debug: Log response status
        print(f"🔍 DEBUG: Response status code: {response.status_code}")
        print(f"🔍 DEBUG: Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        data = response.json()

        # Debug: Log successful response structure
        print(f"🔍 DEBUG: Successful response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

        # The response usually contains choices with message content
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return {"error": "Unexpected API response format", "response": data}

    except requests.exceptions.RequestException as e:
        # Enhanced error handling to show more details
        error_details = {
            "error": f"Perplexity API request failed: {str(e)}",
            "status_code": getattr(e.response, 'status_code', None),
            "request_url": url,
            "request_payload": payload,  # Include the payload that was sent
            "request_headers": {k: v for k, v in headers.items() if k != "Authorization"}  # Exclude API key
        }
        
        # Try to get response text for more details
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details["response_text"] = e.response.text
                error_details["response_headers"] = dict(e.response.headers)
                
                # Try to parse JSON error response
                if e.response.headers.get('content-type', '').startswith('application/json'):
                    error_details["response_json"] = e.response.json()
            except Exception as parse_error:
                error_details["parse_error"] = str(parse_error)
        
        # Debug: Print detailed error information
        print(f"🚨 DEBUG: API Error Details:")
        for key, value in error_details.items():
            if key != "request_payload":  # Don't print payload twice
                print(f"  {key}: {value}")
        
        return error_details


def parse_perplexity_response(response_text):
    """
    Parse Perplexity response to extract structured data fields.
    
    Args:
        response_text (str): Raw response text from Perplexity
        
    Returns:
        dict: Parsed data with extracted fields
    """
    if not response_text or not isinstance(response_text, str):
        return {}
    
    # Enhanced regex patterns for extraction
    patterns = {
        "basic_info": [
            r"Info:\s*(.*?)(?=Total Units:|Average Unit Size:|$)",
            r"Project Info:\s*(.*?)(?=Total Units:|Average Unit Size:|$)",
            r"Description:\s*(.*?)(?=Total Units:|Average Unit Size:|$)"
        ],
        "total_units": [
            r"Total Units:\s*([0-9,\.]+)",
            r"Number of Units:\s*([0-9,\.]+)",
            r"Units:\s*([0-9,\.]+)"
        ],
        "average_unit_size": [
            r"Average Unit Size:\s*([0-9,\.]+)",
            r"Unit Size:\s*([0-9,\.]+)",
            r"Average Size:\s*([0-9,\.]+)"
        ],
        "asp": [
            r"Average Selling Price:\s*([0-9,\.]+)",
            r"Selling Price:\s*([0-9,\.]+)",
            r"Price per sqm:\s*([0-9,\.]+)",
            r"ASP:\s*([0-9,\.]+)"
        ],
        "gfa": [
            r"Gross Floor Area:\s*([0-9,\.]+)",
            r"Floor Area:\s*([0-9,\.]+)",
            r"GFA:\s*([0-9,\.]+)",
            r"Total Floor Area:\s*([0-9,\.]+)"
        ],
        "construction_cost_per_sqm": [
            r"Construction Cost per sqm:\s*([0-9,\.]+)",
            r"Construction Cost:\s*([0-9,\.]+)",
            r"Building Cost per sqm:\s*([0-9,\.]+)"
        ],
        "land_area": [
            r"Land Area:\s*([0-9,\.]+)",
            r"Site Area:\s*([0-9,\.]+)",
            r"Plot Area:\s*([0-9,\.]+)"
        ],
        "land_cost_per_sqm": [
            r"Land Cost per sqm:\s*([0-9,\.]+)",
            r"Land Cost:\s*([0-9,\.]+)",
            r"Land Price per sqm:\s*([0-9,\.]+)"
        ],
        "sources": [
            r"Sources:\s*(.*?)(?=Confidence:|Analysis Method:|$)",
            r"Source:\s*(.*?)(?=Confidence:|Analysis Method:|$)"
        ],
        "confidence": [
            r"Confidence:\s*(.*?)(?=Analysis Method:|\n|$)",
            r"Confidence Level:\s*(.*?)(?=Analysis Method:|\n|$)"
        ],
        "analysis_method": [
            r"Analysis Method:\s*(.*?)(?=Unit Size Analysis:|\n|$)",
            r"Method:\s*(.*?)(?=Unit Size Analysis:|\n|$)"
        ]
    }
    
    result = {}
    
    # Try multiple patterns for each field
    for key, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            m = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if m:
                value = m.group(1).strip()
                
                # Clean up numeric values
                if key not in ["basic_info", "sources", "confidence", "analysis_method"] and value:
                    # Remove all non-numeric characters except dots
                    cleaned_value = re.sub(r'[^\d\.]', '', value)
                    # Handle multiple dots (keep only the first one)
                    if cleaned_value.count('.') > 1:
                        parts = cleaned_value.split('.')
                        cleaned_value = parts[0] + '.' + ''.join(parts[1:])
                    # Remove trailing dots
                    cleaned_value = cleaned_value.rstrip('.')
                    value = cleaned_value
                
                result[key] = value
                found = True
                break
    
    return result