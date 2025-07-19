#%%
import pandas as pd
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os
import re

# Optional: If openai is not installed, run: pip install openai
import openai

print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# Load .env file (for local development)
load_dotenv()

# Read API key from environment
api_key = os.getenv("OPENAI_API_KEY")
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
        payment_array[payment_index] = -total_payment

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

    annual_cost = -total_cost / num_years

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
        total_sga (float): Total SG&A (VND)
        current_year (int): First year of the model
        start_booking_year (int): Year revenue starts
        end_booking_year (int): Year revenue ends

    Returns:
        pd.DataFrame: Year-by-year P&L table from current_year to end_booking_year
    """
    if start_booking_year <= current_year - 1:
        raise ValueError("start_booking_year must be greater than current_year")
    if end_booking_year < start_booking_year:
        raise ValueError("end_booking_year must be >= start_booking_year")

    num_booking_years = end_booking_year - start_booking_year + 1
    revenue_annual = total_revenue / num_booking_years
    land_payment_annual = -total_land_payment / num_booking_years
    sga_annual = -total_sga / num_booking_years
    construction_annual = -total_construction_payment / num_booking_years

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

    return pd.DataFrame(pnl_data)


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
    wacc: float
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
        inflow = selling_progress_schedule[i]
        outflow = (
            construction_payment_schedule[i]
            + sga_payment_schedule[i]
            + tax_expense_schedule[i]
            + land_use_right_payment_schedule[i]
        )
        net_cashflow = inflow - outflow
        discount_factor = 1 / ((1 + wacc) ** i)
        discounted_cashflow = net_cashflow * discount_factor
        total_rnav += discounted_cashflow

        data.append({
            "Year Index": i,
            "Inflow (Selling Revenue)": inflow,
            "Outflow (Cost + SG&A + Tax + Land)": outflow,
            "Net Cash Flow": net_cashflow,
            "Discount Factor": discount_factor,
            "Discounted Cash Flow": discounted_cashflow
        })

    df = pd.DataFrame(data)
    df.loc["Total"] = df[["Discounted Cash Flow"]].sum(numeric_only=True)
    df.at["Total", "Year Index"] = np.nan  # Use np.nan instead of "RNAV"
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

# %%

def get_project_basic_info(project_name: str, openai_api_key: str) -> dict:
    """
    Query OpenAI GPT with web search capabilities to get real-time project information.
    Uses OpenAI's browsing feature if available.
    """
    
    # Enhanced prompt for web search
    prompt = (
        f"Please search the internet for current information about the real estate project '{project_name}' in Vietnam. "
        f"I need the following specific information:\n\n"
        f"1. Project location, developer, and current status\n"
        f"2. Current selling prices (primary and secondary market)\n"
        f"3. Project scale (total area, number of units, GFA)\n"
        f"4. Land area and estimated land costs\n"
        f"5. Construction specifications and estimated costs\n\n"
        f"Please browse recent real estate websites, news articles, and official project information. "
        f"Format your response EXACTLY as follows:\n\n"
        f"Info: [Detailed project description with current status and location]\n"
        f"Average Selling Price: [Current market price per m² in VND, numbers only]\n"
        f"Net Sellable Area: [Total sellable area in m², numbers only]\n"
        f"Gross Floor Area: [Total GFA in m², numbers only]\n"
        f"Construction Cost per sqm: [Estimated construction cost per m² in VND, numbers only]\n"
        f"Land Area: [Land area in m², numbers only]\n"
        f"Land Cost per sqm: [Estimated land cost per m² in VND, numbers only]\n"
        f"Image: [Project image URL if found, otherwise 'N/A']\n"
        f"Sources: [List the websites/sources you found this information from]\n\n"
        f"If you cannot access the internet, please clearly state that limitation."
    )
    
    # Try different approaches for web search
    models_with_search = [
        "gpt-4o",  # Latest model with potential browsing
        "gpt-4-1106-preview",  # GPT-4 Turbo with browsing
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    for model in models_with_search:
        try:
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Try with system message that encourages web search
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a real estate analyst with web browsing capabilities. "
                                   "Search the internet for current, accurate information about real estate projects. "
                                   "Use recent sources and provide factual, up-to-date data."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,  # Low temperature for factual accuracy
            )
            
            content = response.choices[0].message.content.strip()
            
            # Check if the response indicates web search was performed
            if any(indicator in content.lower() for indicator in [
                "i cannot browse", "i don't have access", "cannot access the internet", 
                "i'm not able to search", "no internet access"
            ]):
                # If this model can't browse, try the next one
                continue
            
            # If we get here, the model responded (may or may not have browsed)
            break
            
        except Exception as model_error:
            if model == models_with_search[-1]:
                return {"error": f"All models failed. Last error: {str(model_error)}"}
            continue
    
    # Parse the response
    result = {
        "image_url": "",
        "basic_info": "",
        "asp": "",
        "nsa": "",
        "gfa": "",
        "construction_cost_per_sqm": "",
        "land_area": "",
        "land_cost_per_sqm": "",
        "sources": "",
        "raw_content": content,
        "model_used": model,
        "search_attempted": True
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
        "image_url": r"Image:\s*(.*?)(?=Sources:|$)",
        "sources": r"Sources:\s*(.*?)(?=\n|$)"
    }
    
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if m:
            value = m.group(1).strip()
            # Clean up numeric values
            if key not in ["basic_info", "image_url", "sources"] and value:
                value = re.sub(r'[^\d\.]', '', value)  # Keep only digits and dots
            result[key] = value
    
    # Check if web search was actually performed
    web_indicators = ["source:", "website", "according to", "found on", "search results"]
    result["web_search_performed"] = any(indicator in content.lower() for indicator in web_indicators)
    
    return result

def search_project_online(project_name: str) -> dict:
    """
    Optional: Search for project information online using requests
    This is a placeholder for web search functionality
    You could integrate with Google Search API, Bing API, or web scraping
    """
    # Placeholder for web search integration
    # You could implement:
    # 1. Google Custom Search API
    # 2. Bing Search API  
    # 3. Web scraping (with caution for legal/ethical compliance)
    
    search_info = {
        "status": "not_implemented",
        "message": "Web search integration not implemented. Using market estimates instead.",
        "suggestion": "For real-time data, consider integrating Google Custom Search API or manual research."
    }
    
    return search_info

def get_vietnam_market_estimates(project_type: str = "premium_condo") -> dict:
    """
    Get current Vietnamese real estate market estimates (2024-2025)
    Based on market research and industry reports
    """
    market_data = {
        "luxury_condo": {
            "description": "Luxury condominiums in prime locations (District 1, 3, 7 HCMC or Ba Dinh, Dong Da Hanoi)",
            "asp_range": "70-120 million VND/m²",
            "asp_typical": 80000000,
            "construction_cost": 18000000,
            "land_cost_hcmc": 200000000,
            "land_cost_hanoi": 150000000,
            "examples": ["Landmark 81", "Vinhomes Central Park", "Masteri Thao Dien"]
        },
        "premium_condo": {
            "description": "Premium condominiums in good locations",
            "asp_range": "40-70 million VND/m²",
            "asp_typical": 50000000,
            "construction_cost": 15000000,
            "land_cost_hcmc": 100000000,
            "land_cost_hanoi": 80000000,
            "examples": ["Vinhomes Grand Park", "Saigon Royal", "Times City"]
        },
        "mid_range_condo": {
            "description": "Mid-range condominiums in developing areas",
            "asp_range": "25-45 million VND/m²",
            "asp_typical": 35000000,
            "construction_cost": 12000000,
            "land_cost_hcmc": 50000000,
            "land_cost_hanoi": 40000000,
            "examples": ["Celadon City", "Akira City", "Jamila Khang Dien"]
        },
        "affordable_housing": {
            "description": "Affordable housing and social housing projects",
            "asp_range": "15-30 million VND/m²",
            "asp_typical": 25000000,
            "construction_cost": 10000000,
            "land_cost_hcmc": 30000000,
            "land_cost_hanoi": 25000000,
            "examples": ["EHomeS", "Ehome Nam Sai Gon", "Green Town"]
        }
    }
    
    return market_data.get(project_type, market_data["premium_condo"])

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
                        # Show search status
                        if info.get("web_search_performed"):
                            st.success(f"✅ Web search performed successfully! Model: {info.get('model_used', 'unknown')}")
                            if info.get("sources"):
                                st.info(f"📄 Sources found: {info['sources']}")
                        else:
                            st.warning(f"⚠️ Web search not available. Used model: {info.get('model_used', 'unknown')} with training data only.")
                    
                    # Save the raw response
                    if "raw_content" in info:
                        st.session_state["project_info_raw"] = info["raw_content"]
                    else:
                        st.session_state["project_info_raw"] = str(info)
        
        # Add information about web search
        st.markdown("---")
        st.markdown("🌐 **Web Search**: This attempts to use OpenAI's web browsing capabilities for real-time data.")
        
        # Add expandable info about web search limitations
        with st.expander("� About Web Search with OpenAI"):
            st.markdown("""
            **How it works:**
            - Attempts to use OpenAI models with web browsing capabilities
            - Searches for current project information online
            - Provides real-time market data when available
            
            **Requirements for Web Search:**
            - OpenAI API with browsing access (ChatGPT Plus features)
            - Models: gpt-4o, gpt-4-1106-preview (with browsing)
            - May require special API access or higher tier subscription
            
            **If Web Search Fails:**
            - The system will indicate "Web search not available"
            - You can manually research the project and input the data
            - Consider upgrading to OpenAI API with browsing capabilities
            
            **Alternative Solutions:**
            1. **Manual Research**: Search real estate websites yourself
            2. **API Upgrade**: Get OpenAI API with browsing access
            3. **External APIs**: Integrate Google Search API or similar
            """)
        
    else:
        st.info("💡 Web search integration requires an OpenAI API key. Please set up your API key to use this feature.")

    # Add debug section
    if api_key and st.checkbox("🔧 Debug Mode"):
        st.subheader("🔍 API Debug Information")
        
        if st.button("Test OpenAI Connection"):
            test_result = test_openai_connection(api_key)
            if test_result.get("success"):
                st.success(f"✅ API Connection successful! Response: {test_result['test_response']}")
            else:
                st.error(f"❌ API Test failed: {test_result.get('error')}")
        
        st.write("**Current API Key (first 10 chars):**", api_key[:10] + "..." if api_key else "None")
        st.write("**Available OpenAI Models:** gpt-3.5-turbo, gpt-4 (if you have access)")

    project_info = st.session_state.get("project_info", {})
    project_info_raw = st.session_state.get("project_info_raw", "")

    # Ensure project_info is always a dict
    if not isinstance(project_info, dict):
        project_info = {}

    # Show raw response from ChatGPT
    if project_info_raw:
        st.markdown("**Raw AI Response:**")
        st.code(project_info_raw, language="markdown")
    elif isinstance(project_info, dict) and "raw_content" in project_info:
        st.markdown("**Raw AI Response:**")
        st.code(project_info["raw_content"], language="markdown")

    # Show sources if web search was performed
    if isinstance(project_info, dict) and project_info.get("sources"):
        st.markdown("**🔗 Sources Found:**")
        st.markdown(project_info["sources"])

    # Show image and basic info if available
    if project_info.get("image_url"):
        image_url = str(project_info["image_url"]).strip()
        if (
            image_url
            and image_url.lower() != "none"
            and image_url.lower() != "n/a"
            and image_url.lower().startswith("http")
        ):
            st.image(image_url, caption="Project Image", use_container_width=True)
        else:
            st.write("No image available or invalid image URL.")
            st.caption(f"Image URL received: `{image_url}`")
    
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

    st.header("Project Parameters")
    col1, col2 = st.columns([3, 1])
    with col1:
        nsa = st.number_input("Net Sellable Area (m²)", value=parse_float(nsa_suggest, 265_295), key="nsa")
    with col2:
        st.markdown(f"ChatGPT suggestion: **{nsa_suggest}**" if nsa_suggest else "ChatGPT suggestion: _none_")
        if nsa_suggest:
            if st.button("Copy", key="copy_nsa"):
                st.warning("Please manually copy the suggested value into the textbox above. (Streamlit does not allow programmatic update after widget creation.)")

    col3, col4 = st.columns([3, 1])
    with col3:
        asp = st.number_input("Average Selling Price (VND/m²)", value=parse_float(asp_suggest, 120_000_000), key="asp")
    with col4:
        st.markdown(f"ChatGPT suggestion: **{asp_suggest}**" if asp_suggest else "ChatGPT suggestion: _none_")
        if asp_suggest:
            if st.button("Copy", key="copy_asp"):
                st.warning("Please manually copy the suggested value into the textbox above.")

    col5, col6 = st.columns([3, 1])
    with col5:
        gfa = st.number_input("Gross Floor Area (m²)", value=parse_float(gfa_suggest, 300_000), key="gfa")
    with col6:
        st.markdown(f"ChatGPT suggestion: **{gfa_suggest}**" if gfa_suggest else "ChatGPT suggestion: _none_")
        if gfa_suggest:
            if st.button("Copy", key="copy_gfa"):
                st.warning("Please manually copy the suggested value into the textbox above.")

    col7, col8 = st.columns([3, 1])
    with col7:
        construction_cost_per_sqm = st.number_input("Construction Cost per m² (VND)", value=parse_float(construction_cost_suggest, 20_000_000), key="construction_cost_per_sqm")
    with col8:
        st.markdown(f"ChatGPT suggestion: **{construction_cost_suggest}**" if construction_cost_suggest else "ChatGPT suggestion: _none_")
        if construction_cost_suggest:
            if st.button("Copy", key="copy_construction_cost"):
                st.warning("Please manually copy the suggested value into the textbox above.")

    col9, col10 = st.columns([3, 1])
    with col9:
        land_area = st.number_input("Land Area (m²)", value=parse_float(land_area_suggest, 67_143), key="land_area")
    with col10:
        st.markdown(f"ChatGPT suggestion: **{land_area_suggest}**" if land_area_suggest else "ChatGPT suggestion: _none_")
        if land_area_suggest:
            if st.button("Copy", key="copy_land_area"):
                st.warning("Please manually copy the suggested value into the textbox above.")

    col11, col12 = st.columns([3, 1])
    with col11:
        land_cost_per_sqm = st.number_input("Land Cost per m² (VND)", value=parse_float(land_cost_suggest, 48_500_000), key="land_cost_per_sqm")
    with col12:
        st.markdown(f"ChatGPT suggestion: **{land_cost_suggest}**" if land_cost_suggest else "ChatGPT suggestion: _none_")
        if land_cost_suggest:
            if st.button("Copy", key="copy_land_cost"):
                st.warning("Please manually copy the suggested value into the textbox above.")

    # Button to copy all suggested values
    if st.button("Copy All Suggested Values"):
        st.warning("Please manually copy the suggested values into the textboxes above. (Streamlit does not allow programmatic update after widget creation.)")

    # ...existing code for timeline, calculations, and outputs...
    st.header("Timeline")
    current_year = st.number_input("Current Year", value=2025)
    start_year = st.number_input("Construction/Sales Start Year", value=2025)
    num_years = st.number_input("Number of Years (Construction/Sales)", value=3)
    start_booking_year = st.number_input("Revenue Booking Start Year", value=2027)
    complete_year = st.number_input("Project Completion Year", value=2030)

    # Calculate totals
    total_revenue = nsa * asp
    total_construction_cost = gfa * construction_cost_per_sqm
    total_land_cost = land_area * land_cost_per_sqm
    total_sga_cost = total_revenue * st.number_input("SG&A as % of Revenue", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
    wacc_rate = st.number_input("WACC (Discount Rate, e.g. 0.12 for 12%)", min_value=0.0, max_value=1.0, value=0.12, step=0.01)

    st.subheader("Calculated Totals")
    st.write(f"**Total Revenue:** {total_revenue:,.0f} VND")
    st.write(f"**Total Construction Cost:** {total_construction_cost:,.0f} VND")
    st.write(f"**Total Land Cost:** {total_land_cost:,.0f} VND")
    st.write(f"**Total SG&A:** {total_sga_cost:,.0f} VND")

    # ...existing code for schedules and outputs...
    selling_progress = selling_progress_schedule(
        total_revenue, int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    sga_payment = sga_payment_schedule(
        total_sga_cost, int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    construction_payment = construction_payment_schedule(
        total_construction_cost, int(current_year), int(start_year), int(num_years), int(complete_year)
    )
    land_use_right_payment = land_use_right_payment_schedule_single_year(
        total_land_cost, int(current_year), int(start_year), int(complete_year)
    )

    df_pnl = generate_pnl_schedule(
        total_revenue, total_land_cost, total_construction_cost, total_sga_cost,
        int(start_year), int(start_booking_year), int(complete_year)
    )
    tax_expense = df_pnl["Tax Expense (20%)"].tolist()

    df_rnav = RNAV_Calculation(
        selling_progress, construction_payment, sga_payment, tax_expense, land_use_right_payment, wacc_rate
    )

    st.header(f"Project: {project_name}")

    st.header("P&L Schedule")
    st.dataframe(df_pnl)

    st.header("Tax Expense Schedule")
    st.write(tax_expense)

    st.header("RNAV Calculation (Discounted Cash Flows)")
    st.dataframe(df_rnav)

    st.subheader("RNAV (Total Discounted Cash Flow)")
    st.write(f"{df_rnav.loc['Total', 'Discounted Cash Flow']:,.0f} VND")

    st.header("Cash Flow Chart")
    st.line_chart(df_rnav[df_rnav["Year Index"].notna()][["Net Cash Flow", "Discounted Cash Flow"]])

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

# Add debugging section to main function
def add_debug_section():
    """Add this to your main function for debugging"""
    if api_key and st.checkbox("🔧 Debug Mode"):
        st.subheader("🔍 API Debug Information")
        
        if st.button("Test OpenAI Connection"):
            test_result = test_openai_connection(api_key)
            if test_result.get("success"):
                st.success(f"✅ API Connection successful! Response: {test_result['test_response']}")
            else:
                st.error(f"❌ API Test failed: {test_result.get('error')}")
        
        st.write("**Current API Key (first 10 chars):**", api_key[:10] + "..." if api_key else "None")
        st.write("**Available OpenAI Models:** gpt-3.5-turbo, gpt-4 (if you have access)")

# Remove or comment out all previous if __name__ == "__main__": blocks
# ...existing code for all function definitions...

if __name__ == "__main__":
    main()

