#%%
import pandas as pd
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os
import re


# Optional: If openai is not installed, run: pip install openai
import openai

# Google Search API
try:
    from googleapiclient.discovery import build
    GOOGLE_SEARCH_AVAILABLE = True
except ImportError:
    GOOGLE_SEARCH_AVAILABLE = False
    st.warning("Google API client not installed. Install with: pip install google-api-python-client")

# Import utility functions


# print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# Load .env file (for local development)
load_dotenv()

# Read API keys from environment
api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Debug: Print API key status (without exposing keys)
# print(f"Google API Key present: {bool(google_api_key)}")
# print(f"Google Search Engine ID present: {bool(google_search_engine_id)}")
# print(f"Google API Client available: {GOOGLE_SEARCH_AVAILABLE}")

if not api_key:
    st.warning(
        "⚠️ OPENAI_API_KEY environment variable is not set.\n\n"
        "**For local development:**\n"
        "1. Create a file named `.env` in your project root directory\n"
        "2. Add this line: `OPENAI_API_KEY=your_openai_api_key_here`\n\n"
        "**For Streamlit Cloud deployment:**\n"
        "1. Go to your app settings in Streamlit Cloud\n"
        "2. Add OPENAI_API_KEY as a secret in the 'Secrets' section\n\n"
        "⚡ You can still use the calculator without ChatGPT integration!"
    )
    api_key = None  # Allow app to continue without API key

# Check Google Search API setup
google_search_enabled = bool(google_api_key and google_search_engine_id and GOOGLE_SEARCH_AVAILABLE)
if not google_search_enabled and api_key:
    st.info(
        "🔍 **Google Search Integration**: Set up Google Custom Search for enhanced web search.\n\n"
        "**Setup Steps:**\n"
        "1. Get Google Custom Search API key from Google Cloud Console\n"
        "2. Create a Custom Search Engine and get the Search Engine ID\n"
        "3. Add to your .env file:\n"
        "   - `GOOGLE_API_KEY=your_google_api_key`\n"
        "   - `GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id`"
    )

#%%
def selling_progress_schedule(
    total_revenue: float,
    current_year: int,
    start_year: int,
    num_years: int,
    complete_year: int
) -> list:
    """
    Distribute total revenue evenly over a given number of years (num_years),
    starting from start_year. Output is aligned from current_year to complete_year.

    Args:
        nsa (float): Net sellable area (m²)
        price_per_sqm (float): Selling price per m² (VND)
        current_year (int): Start year of the output array
        start_year (int): Year selling begins
        num_years (int): Number of years selling takes
        complete_year (int): Project completion year

    Returns:
        List[float]: Annual revenue array from current_year to complete_year
    """
    if start_year < current_year:
        raise ValueError("start_year must be >= current_year")
    if complete_year < start_year:
        raise ValueError("complete_year must be >= start_year")
    if (start_year + num_years - 1) > complete_year:
        raise ValueError("Selling period exceeds project completion year")

    
    annual_revenue = total_revenue / num_years

    # Build full year list from current_year to complete_year
    full_years = list(range(current_year, complete_year + 1))

    # Create array with revenue in the selling years only
    revenue_by_year = [
        annual_revenue if start_year <= year < start_year + num_years else 0.0
        for year in full_years
    ]

    return revenue_by_year


# 🔄 Example usage:1
# if __name__ == "__main__":
#     nsa = 120_000          # Net sellable area (m²)
#     price = 50_000_000     # VND/m²
#     current_year = 2025
#     start_year = 2026
#     num_years = 3
#     complete_year = 2030

#     result = selling_progress_schedule(nsa * price, current_year, start_year, num_years, complete_year)

#     for i, value in enumerate(result):
#         print(f"{current_year + i}: {value:,.0f} VND")



def land_use_right_payment_schedule_single_year(
    total_payment: float,
    current_year: int,
    payment_year: int,
    complete_year: int
) -> list:
    """
    Generate land use right payment schedule from current_year to complete_year,
    with the entire payment made in payment_year.

    Args:
        land_area (float): Land area in m².
        price_per_m2 (float): Price per m² in VND.
        current_year (int): Base year of the model.
        payment_year (int): Year payment occurs (must be within range).
        complete_year (int): Final year of the project (inclusive).

    Returns:
        List[float]: Annual cash flow array from current_year to complete_year.
    """
    if payment_year > complete_year:
        raise ValueError("payment_year must be earlier than complete_year")

    
    num_years = complete_year - current_year + 1

    payment_array = [0.0] * num_years

    if payment_year < current_year:
        # Payment year is before the current year, so no payment in the schedule
        pass
    else:
        payment_index = payment_year - current_year
        payment_array[payment_index] = total_payment

    return payment_array

# %%
# 🔄 Example usage:
# if __name__ == "__main__":
#     land_area = 560_000              # in m²
#     price_per_m2 = 5_000_000         # 5 million VND/m²
#     current_year = 2025
#     payment_year = 2026
#     complete_year = 2029

#     result = land_use_right_payment_schedule_single_year(
#         land_area * price_per_m2, current_year, payment_year, complete_year
#     )

#     for i, value in enumerate(result):
#         print(f"{current_year + i}: {value:,.0f} VND")


def construction_payment_schedule(
    total_cost: float,
    current_year: int,
    start_year: int,
    num_years: int,
    complete_year: int
) -> list:
    """
    Distribute total construction cost evenly over num_years starting from start_year.
    Output is aligned from current_year to complete_year.

    Args:
        gfa (float): Gross floor area (m²)
        cost_per_sqm (float): Construction cost per m² (VND)
        current_year (int): First year of the output array
        start_year (int): Year construction begins
        num_years (int): Number of years construction takes
        complete_year (int): Final year of the project

    Returns:
        List[float]: Construction cost per year aligned from current_year to complete_year
    """
    if start_year < current_year:
        raise ValueError("start_year must be >= current_year")
    if complete_year < start_year:
        raise ValueError("complete_year must be >= start_year")
    if (start_year + num_years - 1) > complete_year:
        raise ValueError("Construction period exceeds project completion year")

    annual_cost = total_cost / num_years

    # Create timeline from current_year to complete_year
    full_years = list(range(current_year, complete_year + 1))

    # Allocate cost to construction years
    cost_by_year = [
        annual_cost if start_year <= year < start_year + num_years else 0.0
        for year in full_years
    ]

    return cost_by_year


# 🔄 Example usage:
# if __name__ == "__main__":
#     gfa = 200_000             # m²
#     cost_per_sqm = 20_000_000 # VND/m²
#     current_year = 2025
#     start_year = 2026
#     num_years = 4
#     complete_year = 2030
#     result = construction_payment_schedule(gfa * cost_per_sqm, current_year, start_year, num_years, complete_year)

#     for i, value in enumerate(result):
#         print(f"{current_year + i}: {value:,.0f} VND")



def sga_payment_schedule(
    total_sga: float,
    current_year: int,
    start_year: int,
    num_years: int,
    complete_year: int
) -> list:
    """
    Distribute total revenue evenly over a given number of years (num_years),
    starting from start_year. Output is aligned from current_year to complete_year.

    Args:
        nsa (float): Net sellable area (m²)
        price_per_sqm (float): Selling price per m² (VND)
        current_year (int): Start year of the output array
        start_year (int): Year selling begins
        num_years (int): Number of years selling takes
        complete_year (int): Project completion year

    Returns:
        List[float]: Annual revenue array from current_year to complete_year
    """
    if start_year < current_year:
        raise ValueError("start_year must be >= current_year")
    if complete_year < start_year:
        raise ValueError("complete_year must be >= start_year")
    if (start_year + num_years - 1) > complete_year:
        raise ValueError("Selling period exceeds project completion year")

    
    annual_sga = total_sga / num_years

    # Build full year list from current_year to complete_year
    full_years = list(range(current_year, complete_year + 1))

    # Create array with revenue in the selling years only
    sga_by_year = [
        annual_sga if start_year <= year < start_year + num_years else 0.0
        for year in full_years
    ]

    return sga_by_year

# %%
# 🔄 Example usage:
# if __name__ == "__main__":
#     nsa = 120_000          # Net sellable area (m²)
#     price = 50_000_000     # VND/m²
#     sga_as_percent = 0.08
#     current_year = 2025
#     start_year = 2026
#     num_years = 3
#     complete_year = 2029

#     result = sga_payment_schedule(nsa * price, sga_as_percent, current_year, start_year, num_years, complete_year)

#     for i, value in enumerate(result):
#         print(f"{current_year + i}: {value:,.0f} VND")

# %%

def generate_pnl_schedule(
    total_revenue: float,
    total_land_payment: float,
    total_construction_payment: float,
    total_sga: float,
    current_year: int,
    start_booking_year: int,
    end_booking_year: int
) -> pd.DataFrame:
    """
    Generate a simplified P&L schedule from current_year to end_booking_year.
    All values are 0 for years before start_booking_year.

    Args:
        total_revenue (float): Total revenue (VND)
        total_land_payment (float): Total land use cost (VND)
        total_construction_payment (float): Total construction payment (VND)
        total_sga (float): Total SG&A (VND)
        current_year (int): First year of the model
        start_booking_year (int): Year revenue starts
        end_booking_year (int): Year revenue ends

    Returns:
        pd.DataFrame: Year-by-year P&L table from current_year to end_booking_year with totals row
    """
    if start_booking_year <= current_year - 1:
        raise ValueError("start_booking_year must be greater than current_year")
    if end_booking_year < start_booking_year:
        raise ValueError("end_booking_year must be >= start_booking_year")

    num_booking_years = end_booking_year - start_booking_year + 1
    revenue_annual = total_revenue / num_booking_years
    land_payment_annual = total_land_payment / num_booking_years
    sga_annual = total_sga / num_booking_years
    construction_annual = total_construction_payment / num_booking_years

    pnl_data = []
    for year in range(current_year, end_booking_year + 1):
        if year < start_booking_year:
            revenue = land_cost = sga = pbt = tax = pat = construction = 0.0
        else:
            revenue = revenue_annual
            land_cost = land_payment_annual
            sga = sga_annual
            construction = construction_annual
            pbt = revenue + land_cost + sga + construction_annual
            tax = -pbt * 0.2
            pat = pbt + tax

        pnl_data.append({
            "Year": year,
            "Revenue": revenue,
            "Land Payment": land_cost,
            "Construction": construction,
            "SG&A": sga,
            "Profit Before Tax": pbt,
            "Tax Expense (20%)": tax,
            "Profit After Tax": pat
        })

    df = pd.DataFrame(pnl_data)
    
    # Add total row
    total_row = {
        "Year": "Total",
        "Revenue": df["Revenue"].sum(),
        "Land Payment": df["Land Payment"].sum(),
        "Construction": df["Construction"].sum(),
        "SG&A": df["SG&A"].sum(),
        "Profit Before Tax": df["Profit Before Tax"].sum(),
        "Tax Expense (20%)": df["Tax Expense (20%)"].sum(),
        "Profit After Tax": df["Profit After Tax"].sum()
    }
    
    # Use pd.concat instead of deprecated append
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df


# 🔄 Example usage:
# if __name__ == "__main__":
#     df_pnl = generate_pnl_schedule(
#         total_revenue=10_000_000_000_000,      # 10 trillion VND
#         total_land_payment=3_000_000_000_000,  # 3 trillion VND
#         total_sga=1_000_000_000_000,           # 1 trillion VND
#         current_year=2025,
#         start_booking_year=2026,
#         end_booking_year=2030
#     )

#     print("📄 P&L Schedule:")
#     print(df_pnl)

#     print("\n💰 Tax Expense Schedule:")
#     tax_schedule = df_pnl["Tax Expense (20%)"].tolist()
#     print(tax_schedule)
# %%

def RNAV_Calculation(
    selling_progress_schedule: list,
    construction_payment_schedule: list,
    sga_payment_schedule: list,
    tax_expense_schedule: list,
    land_use_right_payment_schedule: list,
    wacc: float,
    current_year: int
) -> pd.DataFrame:
    """
    Calculate RNAV using discounted cash flow method.

    Args:
        selling_progress_schedule (list of float): Cash inflow per year
        construction_payment_schedule (list of float): Construction cost per year
        sga_payment_schedule (list of float): SG&A cost per year
        tax_expense_schedule (list of float): Tax expense per year
        land_use_right_payment_schedule (list of float): Land use right payments per year
        wacc (float): Discount rate (e.g. 0.12 for 12%)
        current_year (int): Starting year for the analysis

    Returns:
        pd.DataFrame: Year-by-year cash flows and total discounted RNAV
    """
    n = len(selling_progress_schedule)
    # Validate all input lists are of equal length
    all_schedules = [
        construction_payment_schedule,
        sga_payment_schedule,
        tax_expense_schedule,
        land_use_right_payment_schedule
    ]
    if not all(len(lst) == n for lst in all_schedules):
        raise ValueError("All input lists must be of equal length.")

    data = []
    total_rnav = 0.0
    for i in range(n):
        year = current_year + i
        inflow = selling_progress_schedule[i]
        outflow = (
            construction_payment_schedule[i]
            + sga_payment_schedule[i]
            + tax_expense_schedule[i]
            + land_use_right_payment_schedule[i]
        )
        net_cashflow = inflow + outflow
        discount_factor = 1 / ((1 + wacc) ** i)
        discounted_cashflow = net_cashflow * discount_factor
        total_rnav += discounted_cashflow

        data.append({
            "Year": year,
            "Year Index": i,
            "Inflow (Selling Revenue)": inflow,
            "Outflow (Cost + SG&A + Tax + Land)": outflow,
            "Net Cash Flow": net_cashflow,
            "Discount Factor": discount_factor,
            "Discounted Cash Flow": discounted_cashflow
        })

    df = pd.DataFrame(data)
    df.loc["Total"] = df[["Discounted Cash Flow"]].sum(numeric_only=True)
    df.at["Total", "Year"] = "Total"
    df.at["Total", "Year Index"] = np.nan
    return df


# 🔄 Example usage
#%%
# if __name__ == "__main__":
    
#     nsa = 265_295
#     asp = 120_000_000
#     gfa = 300_000
#     construction_cost_per_sqm = 20_000_000
#     land_area = 67_143
#     land_cost_per_sqm = 48_500_000
#     sga_as_percent = 0.08
#     wacc_rate = 0.12

#     current_year = 2025

#     start_year = 2025
#     num_years = 3
#     start_booking_year = 2027
#     complete_year = 2030

#     total_revenue = nsa * asp
#     total_construction_cost = gfa*construction_cost_per_sqm
#     total_land_cost = land_area*land_cost_per_sqm
#     total_sga_cost = total_revenue * sga_as_percent

#     selling_progress = selling_progress_schedule(total_revenue,current_year,start_year,num_years,complete_year)
#     sga_payment = sga_payment_schedule(total_sga_cost,0.08,current_year,start_year,num_years,complete_year)
#     construction_payment = construction_payment_schedule(total_construction_cost,current_year,start_year,num_years,complete_year)
#     land_use_right_payment = land_use_right_payment_schedule_single_year(total_land_cost,current_year,start_year,complete_year)

#     df_pnl = generate_pnl_schedule(total_revenue,total_land_cost,total_construction_cost,total_sga_cost,start_year,start_booking_year, complete_year)

#     tax_expense = df_pnl["Tax Expense (20%)"].tolist()

#     df_rnav = RNAV_Calculation(selling_progress,construction_payment,sga_payment,tax_expense,land_use_right_payment,wacc_rate)

#     print(df_rnav) 

def format_vnd_billions(value: float) -> str:
    """
    Format a VND value into billions with proper formatting.
    
    Args:
        value (float): Raw VND value
        
    Returns:
        str: Formatted string in billions VND (e.g., "31.84 billion VND")
    """
    if value == 0:
        return "0 VND"
    
    # Convert to billions
    billions = value / 1_000_000_000
    
    # Format with appropriate decimal places
    if abs(billions) >= 1000:
        # For very large numbers, show fewer decimals
        return f"{billions:,.0f} billion VND"
    elif abs(billions) >= 100:
        # For hundreds of billions, show 1 decimal
        return f"{billions:,.1f} billion VND"
    elif abs(billions) >= 10:
        # For tens of billions, show 1 decimal
        return f"{billions:,.1f} billion VND"
    elif abs(billions) >= 1:
        # For billions, show 2 decimals
        return f"{billions:,.2f} billion VND"
    else:
        # For less than 1 billion, show in millions
        millions = value / 1_000_000
        if abs(millions) >= 1:
            return f"{millions:,.0f} million VND"
        else:
            # For very small amounts, show in thousands or raw
            if abs(value) >= 1_000:
                thousands = value / 1_000
                return f"{thousands:,.0f} thousand VND"
            else:
                return f"{value:,.0f} VND"


# %%

def get_project_basic_info(project_name: str, openai_api_key: str) -> dict:
    """
    Get project information using Google Search + OpenAI analysis
    This combines real-time web search with AI interpretation
    """
    
    # Step 1: Search Google for current information
    search_results = search_project_online(project_name)
    
    # Step 2: Prepare context from search results
    search_context = ""
    if search_results["status"] == "success" and search_results["results"]:
        search_context = "Here are recent search results about this project:\n\n"
        for i, result in enumerate(search_results["results"][:10], 1):  # Use top 10 results
            search_context += f"{i}. **{result['title']}**\n"
            search_context += f"   Link: {result['link']}\n"
            search_context += f"   Info: {result['snippet']}\n\n"
        search_context += "\nBased on these search results, please extract and analyze the following information:\n"
    else:
        search_context = f"Google Search Status: {search_results['message']}\n"
        search_context += "Please provide analysis based on your training data and Vietnamese market knowledge:\n"
    
    # Step 3: Create enhanced prompt with search context
    prompt = (
        f"Analyze the real estate project '{project_name}' in Vietnam.\n\n"
        f"{search_context}\n"
        f"Please extract and provide the following information:\n\n"
        f"1. Project location, developer, and current status\n"
        f"2. Current selling prices (if mentioned in search results)\n"
        f"3. Project scale and specifications\n"
        f"4. Market positioning and category\n\n"
        f"Format your response EXACTLY as follows:\n\n"
        f"Info: [Detailed project description with current status, location, and developer]\n"
        f"Average Selling Price: [Current price per m² in VND, numbers only - if not found, estimate based on project category]\n"
        f"Net Sellable Area: [Total sellable area in m², numbers only - if not found, estimate based on project scale]\n"
        f"Gross Floor Area: [Total GFA in m², numbers only - if not found, estimate based on project scale]\n"
        f"Construction Cost per sqm: [Estimated construction cost per m² in VND, numbers only]\n"
        f"Land Area: [Land area in m², numbers only - if not found, estimate based on project scale]\n"
        f"Land Cost per sqm: [Estimated land cost per m² in VND, numbers only]\n"
        f"Sources: [List key sources from search results or 'Training data + market estimates']\n"
        f"Confidence: [High/Medium/Low based on available data]\n\n"
        f"Use Vietnamese real estate market standards (2024-2025) for estimates when specific data is not available."
    )
    
    # Step 4: Get OpenAI analysis
    models_to_try = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
    
    for model in models_to_try:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Vietnamese real estate analyst. Analyze search results and provide accurate project information. Use current market data when available, and provide reasonable estimates based on Vietnamese market standards when specific data is missing."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.2,  # Low temperature for factual accuracy
            )
            
            content = response.choices[0].message.content.strip()
            break
            
        except Exception as model_error:
            if model == models_to_try[-1]:
                return {"error": f"All models failed. Last error: {str(model_error)}"}
            continue
    
    # Step 5: Parse the response
    result = {
        "basic_info": "",
        "asp": "",
        "nsa": "",
        "gfa": "",
        "construction_cost_per_sqm": "",
        "land_area": "",
        "land_cost_per_sqm": "",
        "sources": "",
        "confidence": "",
        "raw_content": content,
        "model_used": model,
        "search_results_count": len(search_results.get("results", [])),
        "search_status": search_results["status"],
        "google_search_used": search_results["status"] == "success"
    }
    
    # Enhanced regex patterns
    patterns = {
        "basic_info": r"Info:\s*(.*?)(?=Average Selling Price:|$)",
        "asp": r"Average Selling Price:\s*([0-9,\.]+)",
        "nsa": r"Net Sellable Area:\s*([0-9,\.]+)",
        "gfa": r"Gross Floor Area:\s*([0-9,\.]+)",
        "construction_cost_per_sqm": r"Construction Cost per sqm:\s*([0-9,\.]+)",
        "land_area": r"Land Area:\s*([0-9,\.]+)",
        "land_cost_per_sqm": r"Land Cost per sqm:\s*([0-9,\.]+)",
        "sources": r"Sources:\s*(.*?)(?=Confidence:|$)",
        "confidence": r"Confidence:\s*(.*?)(?=\n|$)"
    }
    
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            value = m.group(1).strip()
            # Clean up numeric values
            if key not in ["basic_info", "sources", "confidence"] and value:
                value = re.sub(r'[^\d\.]', '', value)  # Keep only digits and dots
            result[key] = value
    
    return result

def search_project_online(project_name: str) -> dict:
    """
    Search for project information online using Google Custom Search API
    """
    if not google_search_enabled:
        return {
            "status": "not_configured", 
            "message": f"Google Search API not configured. API Key: {bool(google_api_key)}, Engine ID: {bool(google_search_engine_id)}, Client Available: {GOOGLE_SEARCH_AVAILABLE}",
            "results": []
        }
    
    try:
        # Build the Google Search service
        service = build("customsearch", "v1", developerKey=google_api_key)
        
        # Search queries for Vietnamese real estate
        search_queries = [
            f"{project_name} dự án bất động sản Vietnam",
            f"{project_name} real estate project Vietnam price",
            f"{project_name} apartment condo Vietnam selling price",
            f'"{project_name}" developer Vietnam real estate'
        ]
        
        all_results = []
        
        for i, query in enumerate(search_queries):
            try:
                print(f"Executing search query {i+1}: {query}")
                
                # Execute search with more specific parameters
                result = service.cse().list(
                    q=query,
                    cx=google_search_engine_id,
                    num=3,  # Reduced to 3 to avoid quota issues
                    lr='lang_vi|lang_en',  # Vietnamese and English
                    dateRestrict='y2',  # Results from last 2 years
                    safe='medium',
                    fields='items(title,link,snippet,displayLink)'  # Only get needed fields
                ).execute()
                
                print(f"Search {i+1} returned {len(result.get('items', []))} results")
                
                # Extract useful information
                if 'items' in result:
                    for item in result['items']:
                        all_results.append({
                            'title': item.get('title', ''),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'display_link': item.get('displayLink', ''),
                            'query_used': query
                        })
                        
            except Exception as search_error:
                print(f"Search query {i+1} failed: {str(search_error)}")
                # Continue with next query instead of failing completely
                continue
        
        print(f"Total search results collected: {len(all_results)}")
        
        return {
            "status": "success",
            "message": f"Found {len(all_results)} search results from {len(search_queries)} queries",
            "results": all_results,
            "queries_attempted": len(search_queries),
            "successful_queries": len([r for r in all_results if r])
        }
        
    except Exception as e:
        error_msg = f"Google Search API failed: {str(e)}"
        print(f"Google Search Error: {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "results": [],
            "error_type": type(e).__name__
        }


def main():
    st.title("Real Estate RNAV Calculator")

    # Add project name input
    project_name = st.text_input("Project Name", value="My Project")

    # Set your OpenAI API key if available
    if api_key:
        openai.api_key = api_key

    # Add button to get project info from ChatGPT
    if "project_info" not in st.session_state:
        st.session_state["project_info"] = {}
    if "project_info_raw" not in st.session_state:
        st.session_state["project_info_raw"] = ""
    if "show_raw_response" not in st.session_state:
        st.session_state["show_raw_response"] = False
    
    # Only show ChatGPT button if API key is available
    if api_key:
        if st.button("🔍 Search Project Info (Web Search + AI)"):
            with st.spinner("Searching the web for current project information..."):
                info = get_project_basic_info(project_name, api_key)
                st.session_state["project_info"] = info
                
                # Show detailed feedback
                if isinstance(info, dict):
                    if "error" in info:
                        st.error(f"❌ Error: {info['error']}")
                    else:
                        # Show search status with more details
                        search_status = info.get("search_status", "unknown")
                        search_count = info.get("search_results_count", 0)
                        google_used = info.get("google_search_used", False)
                        
                        if google_used:
                            st.success(f"✅ Google Search performed successfully! Found {search_count} results. Model: {info.get('model_used', 'unknown')}")
                        else:
                            st.warning(f"⚠️ Google Search not available ({search_status}). Used model: {info.get('model_used', 'unknown')} with training data only.")
                        
                        if info.get("sources"):
                            st.info(f"📄 Sources found: {info['sources']}")
                    
                    # Save the raw response
                    if "raw_content" in info:
                        st.session_state["project_info_raw"] = info["raw_content"]
                    else:
                        st.session_state["project_info_raw"] = str(info)
        
        # Show current Google Search status
        if google_search_enabled:
            st.success("✅ Google Search is configured and ready to use!")
        else:
            st.info("🔍 **Google Search Integration**: Set up Google Custom Search for enhanced web search.")
    else:
        st.info("💡 Set up your OpenAI API key to use AI-powered project information search.")

    project_info = st.session_state.get("project_info", {})
    project_info_raw = st.session_state.get("project_info_raw", "")

    # Ensure project_info is always a dict
    if not isinstance(project_info, dict):
        project_info = {}

    # Show sources if web search was performed
    if isinstance(project_info, dict) and project_info.get("sources"):
        st.markdown("**🔗 Sources Found:**")
        st.markdown(project_info["sources"])

    # Toggle button for raw AI response
    if project_info_raw or (isinstance(project_info, dict) and "raw_content" in project_info):
        if st.button("🔍 Show/Hide Raw AI Response"):
            st.session_state["show_raw_response"] = not st.session_state["show_raw_response"]
        
        # Show raw response from ChatGPT only if toggled on
        if st.session_state["show_raw_response"]:
            st.markdown("**Raw AI Response:**")
            if project_info_raw:
                st.code(project_info_raw, language="markdown")
            elif isinstance(project_info, dict) and "raw_content" in project_info:
                st.code(project_info["raw_content"], language="markdown")

    # Show image and basic info if available
    if project_info.get("basic_info"):
        st.info(project_info["basic_info"])
    elif project_info.get("error"):
        st.error(project_info["error"])

    # Helper to parse float or fallback to default
    def parse_float(val, default):
        try:
            return float(str(val).replace(",", "").replace(".", "")) if val else default
        except Exception:
            return default

    # Suggestion values from ChatGPT
    nsa_suggest = project_info.get("nsa", "")
    asp_suggest = project_info.get("asp", "")
    gfa_suggest = project_info.get("gfa", "")
    construction_cost_suggest = project_info.get("construction_cost_per_sqm", "")
    land_area_suggest = project_info.get("land_area", "")
    land_cost_suggest = project_info.get("land_cost_per_sqm", "")

    # Create two parallel columns for Project Parameters and Timeline
    param_col, timeline_col = st.columns(2)

    with param_col:
        st.header("Project Parameters")
        col1, col2 = st.columns([3, 1])
        with col1:
            nsa = st.number_input("Net Sellable Area (m²)", value=parse_float(nsa_suggest, 200_000), key="nsa")
        with col2:
            st.markdown(f"AI suggestion: **{nsa_suggest}**" if nsa_suggest else "AI suggestion: _none_")

        col3, col4 = st.columns([3, 1])
        with col3:
            asp = st.number_input("Average Selling Price (VND/m²)", value=parse_float(asp_suggest, 100_000_000), key="asp")
        with col4:
            st.markdown(f"AI suggestion: **{asp_suggest}**" if asp_suggest else "AI suggestion: _none_")

        col5, col6 = st.columns([3, 1])
        with col5:
            gfa = st.number_input("Gross Floor Area (m²)", value=parse_float(gfa_suggest, 300_000), key="gfa")
        with col6:
            st.markdown(f"AI suggestion: **{gfa_suggest}**" if gfa_suggest else "AI suggestion: _none_")

        col7, col8 = st.columns([3, 1])
        with col7:
            construction_cost_per_sqm = st.number_input("Construction Cost per m² (VND)", value=parse_float(construction_cost_suggest, 20_000_000), key="construction_cost_per_sqm")
        with col8:
            st.markdown(f"AI suggestion: **{construction_cost_suggest}**" if construction_cost_suggest else "AI suggestion: _none_")

        col9, col10 = st.columns([3, 1])
        with col9:
            land_area = st.number_input("Land Area (m²)", value=parse_float(land_area_suggest, 50_000), key="land_area")
        with col10:
            st.markdown(f"AI suggestion: **{land_area_suggest}**" if land_area_suggest else "AI suggestion: _none_")

        col11, col12 = st.columns([3, 1])
        with col11:
            land_cost_per_sqm = st.number_input("Land Cost per m² (VND)", value=parse_float(land_cost_suggest, 50_000_000), key="land_cost_per_sqm")
        with col12:
            st.markdown(f"AI suggestion: **{land_cost_suggest}**" if land_cost_suggest else "AI suggestion: _none_")

        # Button to copy all suggested values
        if st.button("Copy All Suggested Values"):
            st.warning("Please manually copy the suggested values into the textboxes above. (Streamlit does not allow programmatic update after widget creation.)")

    with timeline_col:
        st.header("Timeline")
        current_year = st.number_input("Current Year", value=2025)
        start_year = st.number_input("Construction/Sales Start Year", value=2025)
        num_years = st.number_input("Number of Years (Construction/Sales)", value=3)
        start_booking_year = st.number_input("Revenue Booking Start Year", value=2027)
        complete_year = st.number_input("Project Completion Year", value=2030)
        
        st.markdown("---")
        sga_percent = st.number_input("SG&A as % of Revenue", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
        wacc_rate = st.number_input("WACC (Discount Rate, e.g. 0.12 for 12%)", min_value=0.0, max_value=1.0, value=0.12, step=0.01)

    # Calculate totals
    total_revenue = nsa * asp
    total_construction_cost = -gfa * construction_cost_per_sqm
    total_land_cost = -land_area * land_cost_per_sqm
    total_sga_cost = -total_revenue * sga_percent
    total_estimated_PBT = total_revenue + total_land_cost + total_construction_cost + total_sga_cost
    total_estimated_PAT = total_estimated_PBT * 0.8  # Assuming 20% tax rate

    st.subheader("Calculated Totals")
    st.write(f"**Total Revenue:** {format_vnd_billions(total_revenue)}")
    st.write(f"**Total Construction Cost:** {format_vnd_billions(total_construction_cost)}")
    st.write(f"**Total Land Cost:** {format_vnd_billions(total_land_cost)}")
    st.write(f"**Total SG&A:** {format_vnd_billions(total_sga_cost)}")
    st.write(f"**Total Estimated PBT:** {format_vnd_billions(total_estimated_PBT)}")
    st.write(f"**Total Estimated PAT:** {format_vnd_billions(total_estimated_PAT)}")

    # ...existing code for schedules and outputs...
    selling_progress = selling_progress_schedule(
        total_revenue/(10**9), int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    sga_payment = sga_payment_schedule(
        total_sga_cost/(10**9), int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    construction_payment = construction_payment_schedule(
        total_construction_cost/(10**9), int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    land_use_right_payment = land_use_right_payment_schedule_single_year(
        total_land_cost/(10**9), int(current_year), int(start_year), int(complete_year)
    )

    df_pnl = generate_pnl_schedule(
        total_revenue/(10**9), total_land_cost/(10**9), total_construction_cost/(10**9), total_sga_cost/(10**9),
        int(start_year), int(start_booking_year), int(complete_year)
    )
    tax_expense = df_pnl[df_pnl["Year"] != "Total"]["Tax Expense (20%)"].tolist()

    
    st.header(f"Project: {project_name}")

    #Display selling progress as a list
    st.write("**Selling Progress (Billions VND):**")
    st.write(selling_progress)

    # Display construction schedule as a list
    st.write("**Construction Schedule (Billions VND):**")
    st.write(construction_payment)

    # Display SG&A schedule as a list
    st.write("**SG&A Schedule (Billions VND):**")
    st.write(sga_payment)

    # Display land use right payment schedule as a list
    st.write("**Land Use Right Payment Schedule (Single Year, Billions VND):**")
    st.write(land_use_right_payment)    

    df_rnav = RNAV_Calculation(
        selling_progress, construction_payment, sga_payment, tax_expense, land_use_right_payment, wacc_rate, int(current_year)
    )




    # Create two parallel columns for P&L Schedule and RNAV Calculation
    pnl_col, rnav_col = st.columns(2)
    
    with pnl_col:
        st.header("P&L Schedule")
        st.dataframe(df_pnl)
    
    with rnav_col:
        st.header("RNAV Calculation")
        st.dataframe(df_rnav)

    st.subheader("RNAV (Total Discounted Cash Flow)")
    rnav_value = df_rnav.loc['Total', 'Discounted Cash Flow'] * (10**9)
    st.write(f"**{format_vnd_billions(rnav_value)}**")

    st.header("Cash Flow Chart")
    # Filter out the "Total" row and create chart with years on x-axis
    chart_data = df_rnav[df_rnav["Year"] != "Total"].copy()
    # Ensure Year column is integer for clean display
    chart_data["Year"] = chart_data["Year"].astype(int)
    chart_data = chart_data.set_index("Year")[["Net Cash Flow", "Discounted Cash Flow"]]
    
    # Use plotly for better control over x-axis formatting
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Create plotly chart with clean year formatting
    fig = go.Figure()
    
    # Add Net Cash Flow line
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Net Cash Flow"],
        mode='lines+markers',
        name='Net Cash Flow',
        line=dict(color='blue')
    ))
    
    # Add Discounted Cash Flow line
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["Discounted Cash Flow"],
        mode='lines+markers',
        name='Discounted Cash Flow',
        line=dict(color='red')
    ))
    
    # Update layout for clean year display
    fig.update_layout(
        title="Cash Flow Analysis",
        xaxis_title="Year",
        yaxis_title="Cash Flow (Billions VND)",
        xaxis=dict(
            tickmode='linear',
            tick0=chart_data.index.min(),
            dtick=1,
            tickformat='d'  # Display as integer without decimals
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def test_openai_connection(api_key: str, project_name: str = "Test Project"):
    """
    Test function to debug OpenAI API issues
    """
    if not api_key:
        return {"error": "No API key provided"}
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Simple test prompt
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello, API is working!'"}],
            max_tokens=50
        )
        
        return {
            "success": True,
            "test_response": test_response.choices[0].message.content,
            "model": "gpt-3.5-turbo"
        }
        
    except Exception as e:
        return {"error": f"OpenAI API test failed: {str(e)}"}



if __name__ == "__main__":
    main()

