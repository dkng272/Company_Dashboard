import pandas as pd
import streamlit as st
import numpy as np
from dotenv import load_dotenv
import os
import re
import datetime
import requests
import sys
from pymongo import MongoClient
import certifi

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RNAV Calculator (MongoDB Direct)",
    page_icon="🧮",
    layout="wide"
)

@st.cache_resource
def init_mongodb_connection():
    """Initialize MongoDB connection"""
    try:
        # Get MongoDB connection string from .env file
        connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        
        if not connection_string:
            st.error("❌ MONGODB_CONNECTION_STRING not found in .env file. Please add it to your .env file.")
            return None
        
        # Create MongoDB client with SSL certificate verification
        client = MongoClient(connection_string, tlsCAFile=certifi.where())
        
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"❌ Error connecting to MongoDB: {str(e)}")
        return None

def load_companies_data():
    """Load companies from MongoDB Companies collection"""
    try:
        client = init_mongodb_connection()
        if client is None:
            return pd.DataFrame()
        
        # Get database and collection names
        db_name = 'VietnamStocks'
        collection_name = 'Companies'
        
        # Get database and collection
        db = client.get_database(db_name)
        collection = db.get_collection(collection_name)
        
        # Query all companies
        companies_cursor = collection.find({})
        companies_list = list(companies_cursor)
        
        if not companies_list:
            st.write(f"🔍 DEBUG: No companies found in MongoDB database '{db_name}', collection '{collection_name}'.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(companies_list)
        
        # Remove MongoDB ObjectId if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        st.write(f"🔍 DEBUG: Loaded {len(df)} companies from MongoDB")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading companies data from MongoDB: {str(e)}")
        return pd.DataFrame()

def load_projects_data():
    """Load real estate projects from MongoDB database"""
    try:
        client = init_mongodb_connection()
        if client is None:
            return pd.DataFrame()
        
        # Get database and collection names - using VietnamStocks database
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        # Get database and collection
        db = client.get_database(db_name)
        collection = db.get_collection(collection_name)
        
        # Query all projects
        projects_cursor = collection.find({})
        projects_list = list(projects_cursor)
        
        if not projects_list:
            st.write(f"🔍 DEBUG: No projects found in MongoDB database '{db_name}', collection '{collection_name}'.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_projects = pd.DataFrame(projects_list)
        
        # Remove MongoDB ObjectId if present
        if '_id' in df_projects.columns:
            df_projects = df_projects.drop('_id', axis=1)
        
        st.write(f"🔍 DEBUG: Loaded {len(df_projects)} projects from MongoDB")
        st.write(f"🔍 DEBUG: Project columns: {list(df_projects.columns)}")
        
        # Load companies data and merge
        df_companies = load_companies_data()
        if not df_companies.empty:
            # Merge projects with company information on company_ticker
            df_merged = df_projects.merge(
                df_companies[['ticker', 'company_name', 'sector']], 
                left_on='company_ticker', 
                right_on='ticker', 
                how='left'
            )
            # Drop the duplicate ticker column
            if 'ticker' in df_merged.columns:
                df_merged = df_merged.drop('ticker', axis=1)
            
            # Handle date conversion for last_updated if it exists
            if 'last_updated' in df_merged.columns:
                df_merged['last_updated'] = pd.to_datetime(df_merged['last_updated'], errors='coerce')
            
            st.write(f"🔍 DEBUG: Merged dataframe shape: {df_merged.shape}")
            return df_merged
        else:
            # Handle date conversion for last_updated if it exists
            if 'last_updated' in df_projects.columns:
                df_projects['last_updated'] = pd.to_datetime(df_projects['last_updated'], errors='coerce')
            return df_projects
        
    except Exception as e:
        st.error(f"❌ Error loading projects data from MongoDB: {str(e)}")
        return pd.DataFrame()

def get_companies_list():
    """Get formatted list of companies for selectbox"""
    df_companies = load_companies_data()
    if df_companies.empty:
        return []
    
    # Format as "TICKER - Company Name"
    companies_list = []
    for _, row in df_companies.iterrows():
        ticker = row.get('ticker', '')
        name = row.get('company_name', '')
        if ticker and name:
            companies_list.append(f"{ticker} - {name}")
    
    return sorted(companies_list)

def get_projects_for_company(company_ticker):
    """Get projects for a specific company"""
    df_projects = load_projects_data()
    if df_projects.empty:
        return []
    
    # Filter projects by company ticker
    company_projects = df_projects[df_projects['company_ticker'] == company_ticker]
    if company_projects.empty:
        return []
    
    return sorted(company_projects['project_name'].tolist())

def get_project_data(company_ticker, project_name):
    """Get specific project data"""
    df_projects = load_projects_data()
    if df_projects.empty:
        return None
    
    # Find the specific project
    project_data = df_projects[
        (df_projects['company_ticker'] == company_ticker) & 
        (df_projects['project_name'] == project_name)
    ]
    
    if project_data.empty:
        return None
    
    return project_data.iloc[0].to_dict()

def save_project_to_mongodb(project_data, project_name, rnav_value=None):
    """Save project data to MongoDB"""
    try:
        client = init_mongodb_connection()
        if client is None:
            return {"success": False, "message": "Failed to connect to MongoDB"}
        
        # Get database and collection
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client.get_database(db_name)
        collection = db.get_collection(collection_name)
        
        # Prepare document including location
        document = {
            "project_name": project_name,
            "company_ticker": project_data.get('company_ticker', 'MANUAL'),
            "company_name": project_data.get('company_name', 'Manual Entry'),
            "location": project_data.get('location', ''),  # Include location field
            "total_units": project_data.get('total_units', 0),
            "net_sellable_area": project_data.get('total_units', 0) * project_data.get('average_unit_size', 0),
            "average_unit_size": project_data.get('average_unit_size', 0),
            "average_selling_price": project_data.get('average_selling_price', 0),
            "gross_floor_area": project_data.get('gross_floor_area', 0),
            "land_area": project_data.get('land_area', 0),
            "construction_cost_per_sqm": project_data.get('construction_cost_per_sqm', 0),
            "land_cost_per_sqm": project_data.get('land_cost_per_sqm', 0),
            "construction_start_year": project_data.get('construction_start_year', 2025),
            "sale_start_year": project_data.get('sale_start_year', 2025),
            "land_payment_year": project_data.get('land_payment_year', 2025),
            "construction_years": project_data.get('construction_years', 3),
            "sales_years": project_data.get('sales_years', 3),
            "revenue_booking_start_year": project_data.get('revenue_booking_start_year', 2025),
            "project_completion_year": project_data.get('project_completion_year', 2028),
            "sga_percentage": project_data.get('sga_percentage', 0.08),
            "wacc_rate": project_data.get('wacc_rate', 0.12),
            "rnav_value": rnav_value,
            "last_updated": datetime.datetime.now(),
            "created_date": datetime.datetime.now()
        }
        
        # Check if project exists
        existing = collection.find_one({
            "project_name": project_name,
            "company_ticker": document["company_ticker"]
        })
        
        if existing:
            # Update existing document but preserve created_date and location if not provided
            document["created_date"] = existing.get("created_date", datetime.datetime.now())
            # Preserve existing location if new one is empty
            if not document["location"] and existing.get("location"):
                document["location"] = existing["location"]
            result = collection.replace_one(
                {"_id": existing["_id"]}, 
                document
            )
            action = "updated"
            message = f"✅ Project '{project_name}' updated successfully in MongoDB"
        else:
            # Insert new document
            result = collection.insert_one(document)
            action = "saved"
            message = f"✅ Project '{project_name}' saved successfully to MongoDB"
        
        return {"success": True, "message": message, "action": action}
        
    except Exception as e:
        return {"success": False, "message": f"Error saving to MongoDB: {str(e)}"}

# Read API key from environment
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

if not perplexity_api_key:
    st.warning(
        "⚠️ PERPLEXITY_API_KEY environment variable is not set.\n\n"
        "**For local development:**\n"
        "1. Create a file named `.env` in your project root directory\n"
        "2. Add this line: `PERPLEXITY_API_KEY=your_perplexity_api_key_here`\n\n"
        "**For Streamlit Cloud deployment:**\n"
        "1. Go to your app settings in Streamlit Cloud\n"
        "2. Add PERPLEXITY_API_KEY as a secret in the 'Secrets' section\n\n"
        "⚡ You can still use the calculator without Perplexity integration!"
    )
    perplexity_api_key = None  # Allow app to continue without API key

# ...existing RNAV calculation functions...
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
    Now allows start_year to be before current_year for historical tracking.

    Args:
        total_revenue (float): Total revenue
        current_year (int): Start year of the output array
        start_year (int): Year selling begins (can be < current_year)
        num_years (int): Number of years selling takes
        complete_year (int): Project completion year

    Returns:
        List[float]: Annual revenue array from current_year to complete_year
    """
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

def land_use_right_payment_schedule_single_year(
    total_payment: float,
    current_year: int,
    payment_year: int,
    complete_year: int
) -> list:
    """
    Generate land use right payment schedule from current_year to complete_year,
    with the entire payment made in payment_year.
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
    Now allows start_year to be before current_year for historical tracking.
    """
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

def sga_payment_schedule(
    total_sga: float,
    current_year: int,
    start_year: int,
    num_years: int,
    complete_year: int
) -> list:
    """
    Distribute total SG&A evenly over a given number of years (num_years),
    starting from start_year. Output is aligned from current_year to complete_year.
    Now allows start_year to be before current_year for historical tracking.
    """
    if complete_year < start_year:
        raise ValueError("complete_year must be >= start_year")
    if (start_year + num_years - 1) > complete_year:
        raise ValueError("SG&A period exceeds project completion year")

    annual_sga = total_sga / num_years

    # Build full year list from current_year to complete_year
    full_years = list(range(current_year, complete_year + 1))

    # Create array with SG&A in the active years only
    sga_by_year = [
        annual_sga if start_year <= year < start_year + num_years else 0.0
        for year in full_years
    ]

    return sga_by_year

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
    Generate a simplified P&L schedule from start_booking_year to end_booking_year.
    Shows both historical and future data with proper labeling.
    """
    if end_booking_year < start_booking_year:
        raise ValueError("end_booking_year must be >= start_booking_year")

    # Calculate annual amounts based on total booking period
    total_booking_years = end_booking_year - start_booking_year + 1
    revenue_annual = total_revenue / total_booking_years if total_booking_years > 0 else 0
    land_payment_annual = total_land_payment / total_booking_years if total_booking_years > 0 else 0
    sga_annual = total_sga / total_booking_years if total_booking_years > 0 else 0
    construction_annual = total_construction_payment / total_booking_years if total_booking_years > 0 else 0

    pnl_data = []
    for year in range(start_booking_year, end_booking_year + 1):
        # Determine if this is historical or future
        is_historical = year < current_year
        year_type = "Historical" if is_historical else "Future"
        
        # All years in the booking period have values
        revenue = revenue_annual
        land_cost = land_payment_annual
        sga = sga_annual
        construction = construction_annual
        pbt = revenue + land_cost + sga + construction
        tax = -pbt * 0.2 if pbt > 0 else 0.0
        pat = pbt + tax

        pnl_data.append({
            "Year": year,
            "Type": year_type,
            "Revenue": revenue,
            "Land Payment": land_cost,
            "Construction": construction,
            "SG&A": sga,
            "Profit Before Tax": pbt,
            "Tax Expense (20%)": tax,
            "Profit After Tax": pat
        })

    df = pd.DataFrame(pnl_data)
    
    # Add total rows for historical, future, and overall
    historical_df = df[df["Type"] == "Historical"]
    future_df = df[df["Type"] == "Future"]
    
    # Add subtotals
    if not historical_df.empty:
        historical_total = {
            "Year": "Total (Historical)",
            "Type": "Summary",
            "Revenue": historical_df["Revenue"].sum(),
            "Land Payment": historical_df["Land Payment"].sum(),
            "Construction": historical_df["Construction"].sum(),
            "SG&A": historical_df["SG&A"].sum(),
            "Profit Before Tax": historical_df["Profit Before Tax"].sum(),
            "Tax Expense (20%)": historical_df["Tax Expense (20%)"].sum(),
            "Profit After Tax": historical_df["Profit After Tax"].sum()
        }
        df = pd.concat([df, pd.DataFrame([historical_total])], ignore_index=True)
    
    if not future_df.empty:
        future_total = {
            "Year": "Total (Future)",
            "Type": "Summary",
            "Revenue": future_df["Revenue"].sum(),
            "Land Payment": future_df["Land Payment"].sum(),
            "Construction": future_df["Construction"].sum(),
            "SG&A": future_df["SG&A"].sum(),
            "Profit Before Tax": future_df["Profit Before Tax"].sum(),
            "Tax Expense (20%)": future_df["Tax Expense (20%)"].sum(),
            "Profit After Tax": future_df["Profit After Tax"].sum()
        }
        df = pd.concat([df, pd.DataFrame([future_total])], ignore_index=True)
    
    # Add overall total
    overall_total = {
        "Year": "Total (Overall)",
        "Type": "Summary",
        "Revenue": df[df["Type"] != "Summary"]["Revenue"].sum(),
        "Land Payment": df[df["Type"] != "Summary"]["Land Payment"].sum(),
        "Construction": df[df["Type"] != "Summary"]["Construction"].sum(),
        "SG&A": df[df["Type"] != "Summary"]["SG&A"].sum(),
        "Profit Before Tax": df[df["Type"] != "Summary"]["Profit Before Tax"].sum(),
        "Tax Expense (20%)": df[df["Type"] != "Summary"]["Tax Expense (20%)"].sum(),
        "Profit After Tax": df[df["Type"] != "Summary"]["Profit After Tax"].sum()
    }
    df = pd.concat([df, pd.DataFrame([overall_total])], ignore_index=True)

    return df

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
    Only includes future cash flows (current year and beyond) for RNAV calculation.
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
        
        # Break down outflow components
        construction_cost = construction_payment_schedule[i]
        sga_cost = sga_payment_schedule[i]
        tax_cost = tax_expense_schedule[i]
        land_cost = land_use_right_payment_schedule[i]
        
        total_outflow = construction_cost + sga_cost + tax_cost + land_cost
        net_cashflow = inflow + total_outflow
        
        # Calculate discount factor (year 0 = current year)
        discount_factor = 1 / ((1 + wacc) ** i)
        discounted_cashflow = net_cashflow * discount_factor
        
        # Only add to RNAV if it's current year or future (i >= 0)
        total_rnav += discounted_cashflow

        # Determine if this is a future cash flow for RNAV
        included_in_rnav = True  # Since we start from current_year, all are included

        data.append({
            "Year": year,
            "Year Index": i,
            "Inflow (Revenue)": inflow,
            "Construction Cost": construction_cost,
            "Land Cost": land_cost,
            "SG&A": sga_cost,
            "Tax": tax_cost,
            "Total Outflow": total_outflow,
            "Net Cash Flow": net_cashflow,
            "Discount Factor": discount_factor,
            "Discounted Cash Flow": discounted_cashflow,
            "Included in RNAV": "Yes" if included_in_rnav else "No"
        })

    df = pd.DataFrame(data)
    
    # Add total row
    total_row = {
        "Year": "Total RNAV",
        "Year Index": np.nan,
        "Inflow (Revenue)": df["Inflow (Revenue)"].sum(),
        "Construction Cost": df["Construction Cost"].sum(),
        "Land Cost": df["Land Cost"].sum(),
        "SG&A": df["SG&A"].sum(),
        "Tax": df["Tax"].sum(),
        "Total Outflow": df["Total Outflow"].sum(),
        "Net Cash Flow": df["Net Cash Flow"].sum(),
        "Discount Factor": np.nan,
        "Discounted Cash Flow": total_rnav,
        "Included in RNAV": "Summary"
    }
    
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    return df

def format_vnd_billions(value: float) -> str:
    """Format a VND value into billions with proper formatting."""
    if value == 0:
        return "0 VND"
    
    # Convert to billions
    billions = value / 1_000_000_000
    
    # Format with appropriate decimal places
    if abs(billions) >= 1000:
        return f"{billions:,.0f} billion VND"
    elif abs(billions) >= 100:
        return f"{billions:,.1f} billion VND"
    elif abs(billions) >= 10:
        return f"{billions:,.1f} billion VND"
    elif abs(billions) >= 1:
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

def format_number_with_commas(value: str) -> str:
    """Format a numeric string with commas for better readability."""
    if not value or value == "":
        return "none"
    
    try:
        # Convert to float first to handle any decimal points
        num = float(str(value).replace(",", ""))
        # Format with commas
        if num == int(num):
            return f"{int(num):,}"
        else:
            return f"{num:,.2f}"
    except (ValueError, TypeError):
        return str(value)

def main():
    st.title("🧮 Real Estate RNAV Calculator - MongoDB Direct Edition")

    # MongoDB connection status
    with st.sidebar:
        st.header("🔗 Database Connection")
        client = init_mongodb_connection()
        if client:
            st.success("✅ Connected to MongoDB")
            st.info(f"📊 Database: `VietnamStocks`")
            st.info(f"📁 Collections: `Companies`, `RealEstateProjects`")
        else:
            st.error("❌ MongoDB connection failed")
            st.info("Using manual entry mode only")

    # Get current calendar year automatically
    current_calendar_year = datetime.datetime.now().year

    # Check for pre-loaded project data from dashboard
    preload_data = st.session_state.get('preload_project_data', None)
    preload_name = st.session_state.get('preload_project_name', None)

    # Load projects database only if available
    if client:
        st.write("🔍 DEBUG: Loading projects database...")
        df_projects = load_projects_data()
        st.write(f"🔍 DEBUG: Projects database shape: {df_projects.shape}")
        
        # Test companies loading
        st.write("🔍 DEBUG: Loading companies...")
        companies = get_companies_list()
        st.write(f"🔍 DEBUG: Companies loaded: {len(companies)} companies")
        st.write(f"🔍 DEBUG: Sample companies: {companies[:3] if companies else 'No companies found'}")
    else:
        df_projects = pd.DataFrame()
        companies = []
        st.write("🔍 DEBUG: MongoDB not available, using empty DataFrame")
    
    # Project selection interface
    st.header("📋 Project Selection")
    
    if df_projects.empty:
        if client:
            st.warning("No projects found in MongoDB. You can still enter project details manually below.")
            st.write("🔍 DEBUG: DataFrame is empty - check database connection and collection names")
        else:
            st.info("MongoDB not available. Enter project details manually below.")
        project_name = st.text_input("Project Name", value="My Project")
        selected_project_data = None
    else:
        # Check if we have preloaded data to set default selection
        default_company_index = 0
        default_project_index = 0
        
        if preload_data and 'company_ticker' in preload_data:
            preload_company = f"{preload_data['company_ticker']} - {preload_data['company_name']}"
            if preload_company in companies:
                default_company_index = companies.index(preload_company) + 1
        
        selected_company = st.selectbox(
            "Select Company:",
            options=["Select a company..."] + companies,
            index=default_company_index
        )
        
        if selected_company == "Select a company...":
            st.info("👆 Please select a company to see available projects")
            project_name = st.text_input("Or enter project name manually:", value="My Project")
            selected_project_data = None
        else:
            # Extract ticker from selection
            company_ticker = selected_company.split(" - ")[0]
            company_name = selected_company.split(" - ")[1]
            
            # Project selection
            projects = get_projects_for_company(company_ticker)
            
            if not projects:
                st.warning(f"No projects found for {selected_company}")
                project_name = st.text_input("Enter new project name:", value="New Project")
                selected_project_data = None
            else:
                # Check if we have preloaded project to set default
                if preload_name and preload_name in projects:
                    default_project_index = projects.index(preload_name) + 1
                
                selected_project = st.selectbox(
                    "Select Project:",
                    options=["Select a project..."] + projects,
                    index=default_project_index
                )
                
                if selected_project == "Select a project...":
                    st.info("👆 Please select a project or enter a new one")
                    project_name = st.text_input("Or enter new project name:", value="New Project")
                    selected_project_data = None
                else:
                    project_name = selected_project
                    selected_project_data = get_project_data(company_ticker, selected_project)
                    
                    if selected_project_data:
                        st.success(f"✅ Loaded project from MongoDB: {selected_project}")
                        
                        # Show project summary with Location
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Units", f"{int(selected_project_data['total_units']):,}")
                            st.metric("NSA", f"{int(selected_project_data['net_sellable_area']):,} m²")
                        with col2:
                            st.metric("ASP", f"{int(selected_project_data['average_selling_price']/1_000_000):,}M VND/m²")
                            st.metric("GFA", f"{int(selected_project_data['gross_floor_area']):,} m²")
                        with col3:
                            st.metric("Land Area", f"{int(selected_project_data['land_area']):,} m²")
                            st.metric("Completion", f"{int(selected_project_data['project_completion_year'])}")
                        with col4:
                            # Display Location information
                            location = selected_project_data.get('location', 'N/A')
                            st.metric("📍 Location", location if location and location != 'N/A' else "Not specified")
                            # Show RNAV if available
                            if 'rnav_value' in selected_project_data and selected_project_data['rnav_value']:
                                rnav_formatted = format_vnd_billions(selected_project_data['rnav_value'])
                                st.metric("🏆 Stored RNAV", rnav_formatted)
                            else:
                                st.metric("🏆 RNAV Status", "Not calculated")
                        
                        # Show additional project details including location
                        with st.expander("📋 Project Details", expanded=False):
                            detail_col1, detail_col2 = st.columns(2)
                            with detail_col1:
                                st.write(f"**📍 Location:** {location}")
                                st.write(f"**🏗️ Construction Start:** {selected_project_data.get('construction_start_year', 'N/A')}")
                                st.write(f"**📅 Project Completion:** {selected_project_data.get('project_completion_year', 'N/A')}")
                                st.write(f"**🔨 Construction Years:** {selected_project_data.get('construction_years', 'N/A')}")
                            with detail_col2:
                                st.write(f"**💰 Land Cost/m²:** {format_vnd_billions(selected_project_data.get('land_cost_per_sqm', 0))}")
                                st.write(f"**🏗️ Construction Cost/m²:** {format_vnd_billions(selected_project_data.get('construction_cost_per_sqm', 0))}")
                                st.write(f"**📊 SG&A %:** {selected_project_data.get('sga_percentage', 0.08):.1%}")
                                st.write(f"**📈 WACC Rate:** {selected_project_data.get('wacc_rate', 0.12):.1%}")
                        
                        # Override preload_data with database data
                        preload_data = selected_project_data
                    else:
                        st.error("Error loading project data from MongoDB")

    # Helper function for float parsing with preload and fallback
    def parse_float_with_preload(ai_suggest, preload_value, default):
        try:
            if preload_value is not None:
                return float(preload_value)
            elif ai_suggest:
                return float(str(ai_suggest).replace(",", "").replace(".", ""))
            else:
                return float(default)
        except Exception:
            return float(default)

    # Helper function for int parsing with preload and fallback
    def parse_int_with_preload(ai_suggest, preload_value, default):
        try:
            if preload_value is not None:
                return int(preload_value)
            elif ai_suggest:
                return int(str(ai_suggest).replace(",", "").replace(".", ""))
            else:
                return int(default)
        except Exception:
            return int(default)

    # Create two parallel columns for Project Parameters and Timeline
    param_col, timeline_col = st.columns(2)

    with param_col:
        st.header("Project Parameters")
        
        # Add Location input field at the top
        location = st.text_input(
            "📍 Project Location",
            value=preload_data.get('location', '') if preload_data else '',
            placeholder="Enter project location (e.g., District 1, Ho Chi Minh City)",
            key="location"
        )
        if preload_data and 'location' in preload_data and preload_data['location']:
            st.caption(f"📊 From database: **{preload_data['location']}**")
        else:
            st.caption("💡 Enter project location for better documentation")
        
        st.markdown("---")  # Add separator after location
        
        # Total Units
        total_units = st.number_input(
            "Total Units", 
            value=parse_float_with_preload(
                "",
                preload_data.get('total_units') if preload_data else None, 
                2500
            ), 
            key="total_units"
        )
        if preload_data and 'total_units' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['total_units'])))}** units")
        else:
            st.caption("💡 Using default value")
        
        # Average Unit Size
        average_unit_size = st.number_input(
            "Average Unit Size (m²)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('average_unit_size') if preload_data else None, 
                80
            ), 
            key="average_unit_size"
        )
        if preload_data and 'average_unit_size' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['average_unit_size'])))}** m²")
        else:
            st.caption("💡 Using default value")
        
        # Calculate NSA from units and unit size
        nsa = total_units * average_unit_size
        st.info(f"📊 **Calculated Net Sellable Area:** {format_number_with_commas(str(int(nsa)))} m²")
        
        # Average Selling Price
        asp = st.number_input(
            "Average Selling Price (VND/m²)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('average_selling_price') if preload_data else None, 
                100_000_000
            ), 
            key="asp"
        )
        if preload_data and 'average_selling_price' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['average_selling_price'])))}** VND/m²")
        else:
            st.caption("💡 Using default value")
        
        # Gross Floor Area
        gfa = st.number_input(
            "Gross Floor Area (m²)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('gross_floor_area') if preload_data else None, 
                300_000
            ), 
            key="gfa"
        )
        if preload_data and 'gross_floor_area' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['gross_floor_area'])))}** m²")
        else:
            st.caption("💡 Using default value")
        
        # Construction Cost per sqm
        construction_cost_per_sqm = st.number_input(
            "Construction Cost per m² (VND)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('construction_cost_per_sqm') if preload_data else None, 
                20_000_000
            ), 
            key="construction_cost_per_sqm"
        )
        if preload_data and 'construction_cost_per_sqm' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['construction_cost_per_sqm'])))}** VND/m²")
        else:
            st.caption("💡 Using default value")
        
        # Land Area
        land_area = st.number_input(
            "Land Area (m²)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('land_area') if preload_data else None, 
                50_000
            ), 
            key="land_area"
        )
        if preload_data and 'land_area' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['land_area'])))}** m²")
        else:
            st.caption("💡 Using default value")
        
        # Land Cost per sqm
        land_cost_per_sqm = st.number_input(
            "Land Cost per m² (VND)", 
            value=parse_float_with_preload(
                "",
                preload_data.get('land_cost_per_sqm') if preload_data else None, 
                50_000_000
            ), 
            key="land_cost_per_sqm"
        )
        if preload_data and 'land_cost_per_sqm' in preload_data:
            st.caption(f"📊 From database: **{format_number_with_commas(str(int(preload_data['land_cost_per_sqm'])))}** VND/m²")
        else:
            st.caption("💡 Using default value")

    with timeline_col:
        st.header("Timeline")
        
        # Remove the current year input and use calendar year automatically
        current_year = current_calendar_year
        st.info(f"📅 **Current Year:** {current_year} (automatically set to calendar year)")
        
        start_year = st.number_input(
            "Construction/Sales Start Year", 
            value=parse_int_with_preload("", preload_data.get('construction_start_year') if preload_data else None, current_year),
            min_value=current_year - 10,  # Allow up to 10 years in the past
            max_value=current_year + 20   # Allow up to 20 years in the future
        )
        
        # Show warning if start year is in the past
        if start_year < current_year:
            years_ago = current_year - start_year
            st.warning(f"⚠️ Start year is {years_ago} year(s) in the past. Historical data will be shown but not included in RNAV calculation.")
        
        # Add land payment year input
        land_payment_year = st.number_input(
            "Land Payment Year", 
            value=parse_int_with_preload("", preload_data.get('land_payment_year') if preload_data else None, start_year),
            min_value=current_year - 10,  # Allow up to 10 years in the past
            max_value=current_year + 20   # Allow up to 20 years in the future
        )
        
        # Separate construction and sales duration
        construction_years = st.number_input(
            "Number of Years for Construction", 
            value=parse_int_with_preload("", preload_data.get('construction_years') if preload_data else None, 3), 
            min_value=1
        )
        sales_years = st.number_input(
            "Number of Years for Sales", 
            value=parse_int_with_preload("", preload_data.get('sales_years') if preload_data else None, 3), 
            min_value=1
        )
        
        start_booking_year = st.number_input(
            "Revenue Booking Start Year", 
            value=parse_int_with_preload("", preload_data.get('revenue_booking_start_year') if preload_data else None, max(current_year, start_year + 1)),
            min_value=start_year
        )
        complete_year = st.number_input(
            "Project Completion Year", 
            value=parse_int_with_preload("", preload_data.get('project_completion_year') if preload_data else None, start_year + 5),
            min_value=start_year + 1
        )
        
        # Show timeline summary
        st.markdown("---")
        st.markdown("**📊 Project Timeline Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"• Construction: {start_year} - {start_year + construction_years - 1}")
            st.write(f"• Sales: {start_year} - {start_year + sales_years - 1}")
            st.write(f"• Land Payment: {land_payment_year}")
        with col2:
            st.write(f"• Revenue Booking: {start_booking_year} - {complete_year}")
            st.write(f"• Project Duration: {complete_year - start_year + 1} years")
        
        st.markdown("---")
        sga_percent = st.number_input(
            "SG&A as % of Revenue", 
            min_value=0.0, max_value=1.0, 
            value=float(preload_data.get('sga_percentage', 0.08)) if preload_data and 'sga_percentage' in preload_data else 0.08, 
            step=0.01
        )
        wacc_rate = st.number_input(
            "WACC (Discount Rate, e.g. 0.12 for 12%)", 
            min_value=0.0, max_value=1.0, 
            value=float(preload_data.get('wacc_rate', 0.12)) if preload_data and 'wacc_rate' in preload_data else 0.12, 
            step=0.01
        )

    # Add Save Project button only if MongoDB is available
    if client:
        st.sidebar.markdown("---")
        if st.sidebar.button("💾 Save to MongoDB", type="primary"):
            # Calculate RNAV value before saving
            try:
                # Calculate totals
                total_revenue = nsa * asp
                total_construction_cost = -gfa * construction_cost_per_sqm
                total_land_cost = -land_area * land_cost_per_sqm
                total_sga_cost = -total_revenue * sga_percent

                # Generate schedules
                selling_progress = selling_progress_schedule(
                    total_revenue/(10**9), int(current_year), int(start_year), int(sales_years), int(complete_year)
                )
                sga_payment = sga_payment_schedule(
                    total_sga_cost/(10**9), int(current_year), int(start_year), int(sales_years), int(complete_year)
                )
                construction_payment = construction_payment_schedule(
                    total_construction_cost/(10**9), int(current_year), int(start_year), int(construction_years), int(complete_year)
                )
                land_use_right_payment = land_use_right_payment_schedule_single_year(
                    total_land_cost/(10**9), int(current_year), int(land_payment_year), int(complete_year)
                )

                df_pnl = generate_pnl_schedule(
                    total_revenue/(10**9), total_land_cost/(10**9), total_construction_cost/(10**9), total_sga_cost/(10**9),
                    int(current_year), int(start_booking_year), int(complete_year)
                )
                
                # Create tax expense schedule
                num_years = int(complete_year) - int(current_year) + 1
                tax_expense = []
                for year in range(int(current_year), int(complete_year) + 1):
                    year_data = df_pnl[df_pnl["Year"] == year]
                    if not year_data.empty and year_data["Type"].iloc[0] != "Summary":
                        tax_value = year_data["Tax Expense (20%)"].iloc[0]
                    else:
                        tax_value = 0.0
                    tax_expense.append(tax_value)

                # Calculate RNAV
                df_rnav = RNAV_Calculation(
                    selling_progress, construction_payment, sga_payment, tax_expense, land_use_right_payment, wacc_rate, int(current_year)
                )

                # Get RNAV value
                total_row = df_rnav[df_rnav["Year"] == "Total RNAV"]
                if not total_row.empty:
                    rnav_value = total_row["Discounted Cash Flow"].iloc[0] * (10**9)
                else:
                    rnav_value = df_rnav.loc[df_rnav.index[-1], 'Discounted Cash Flow'] * (10**9)

            except Exception as e:
                st.sidebar.error(f"Error calculating RNAV: {str(e)}")
                rnav_value = None

            # Determine company info
            if selected_project_data:
                # Use existing company info from selected project
                company_ticker = selected_project_data.get('company_ticker', 'MANUAL')
                company_name = selected_project_data.get('company_name', 'Manual Entry')
            elif preload_data:
                # Use preload company info
                company_ticker = preload_data.get('company_ticker', 'MANUAL')
                company_name = preload_data.get('company_name', 'Manual Entry')
            else:
                # Default for manual entries
                company_ticker = 'MANUAL'
                company_name = 'Manual Entry'
            
            # Collect current project data including location
            current_project_data = {
                'company_ticker': company_ticker,
                'company_name': company_name,
                'location': location,  # Include location in saved data
                'total_units': total_units,
                'average_unit_size': average_unit_size,
                'average_selling_price': asp,
                'gross_floor_area': gfa,
                'land_area': land_area,
                'construction_cost_per_sqm': construction_cost_per_sqm,
                'land_cost_per_sqm': land_cost_per_sqm,
                'construction_start_year': start_year,
                'sale_start_year': start_year,
                'land_payment_year': land_payment_year,
                'construction_years': construction_years,
                'sales_years': sales_years,
                'revenue_booking_start_year': start_booking_year,
                'project_completion_year': complete_year,
                'sga_percentage': sga_percent,
                'wacc_rate': wacc_rate
            }
            
            save_result = save_project_to_mongodb(current_project_data, project_name, rnav_value)
            if save_result["success"]:
                if rnav_value is not None:
                    st.sidebar.success(f"{save_result['message']}\n💰 RNAV: {format_vnd_billions(rnav_value)}")
                else:
                    st.sidebar.success(save_result["message"])
                if save_result["action"] == "saved":
                    st.sidebar.info("💡 Project saved to MongoDB. Refresh to see in dropdown.")
                    # Refresh the database after save
                    st.rerun()
            else:
                st.sidebar.error(save_result["message"])

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

    # Update schedule calculations to use separate construction and sales years
    selling_progress = selling_progress_schedule(
        total_revenue/(10**9), int(current_year), int(start_year), int(sales_years), int(complete_year)
    )
    sga_payment = sga_payment_schedule(
        total_sga_cost/(10**9), int(current_year), int(start_year), int(sales_years), int(complete_year)
    )
    construction_payment = construction_payment_schedule(
        total_construction_cost/(10**9), int(current_year), int(start_year), int(construction_years), int(complete_year)
    )
    land_use_right_payment = land_use_right_payment_schedule_single_year(
        total_land_cost/(10**9), int(current_year), int(land_payment_year), int(complete_year)
    )

    df_pnl = generate_pnl_schedule(
        total_revenue/(10**9), total_land_cost/(10**9), total_construction_cost/(10**9), total_sga_cost/(10**9),
        int(current_year), int(start_booking_year), int(complete_year)
    )
    
    # Create tax expense schedule that matches the time period from current_year to complete_year
    num_years = int(complete_year) - int(current_year) + 1
    
    # Get tax expense for each year from current_year to complete_year
    tax_expense = []
    for year in range(int(current_year), int(complete_year) + 1):
        # Find tax expense for this year in the P&L schedule
        year_data = df_pnl[df_pnl["Year"] == year]
        if not year_data.empty and year_data["Type"].iloc[0] != "Summary":
            tax_value = year_data["Tax Expense (20%)"].iloc[0]
        else:
            # If year not found in P&L (e.g., before booking period), use 0
            tax_value = 0.0
        tax_expense.append(tax_value)
    
    # Verify all schedules have the same length
    schedules_info = {
        "selling_progress": len(selling_progress),
        "construction_payment": len(construction_payment), 
        "sga_payment": len(sga_payment),
        "tax_expense": len(tax_expense),
        "land_use_right_payment": len(land_use_right_payment)
    }
    
    # Debug information
    st.sidebar.write("Schedule lengths:", schedules_info)
    
    # Ensure all schedules have the same length
    expected_length = num_years
    if not all(length == expected_length for length in schedules_info.values()):
        st.error(f"Schedule length mismatch! Expected: {expected_length}, Got: {schedules_info}")
        st.stop()

    st.header(f"Project: {project_name}")
    
    # Show project location prominently
    if location and location.strip():
        st.info(f"📍 **Location:** {location}")

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
    
    # Get RNAV value from the total row
    try:
        total_row = df_rnav[df_rnav["Year"] == "Total RNAV"]
        if not total_row.empty:
            rnav_value = total_row["Discounted Cash Flow"].iloc[0] * (10**9)
        else:
            # Fallback to old method
            rnav_value = df_rnav.loc[df_rnav.index[-1], 'Discounted Cash Flow'] * (10**9)
    except:
        rnav_value = 0
    
    st.write(f"**{format_vnd_billions(rnav_value)}**")
    
    # Show RNAV history if available
    if selected_project_data and 'rnav_value' in selected_project_data and selected_project_data['rnav_value'] is not None:
        stored_rnav = selected_project_data['rnav_value']
        last_updated = selected_project_data.get('last_updated', 'Unknown')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current RNAV", format_vnd_billions(rnav_value))
        with col2:
            st.metric(
                "Stored RNAV", 
                format_vnd_billions(stored_rnav),
                delta=format_vnd_billions(rnav_value - stored_rnav)
            )
        
        st.caption(f"📅 Last stored: {last_updated}")
    
    if start_year < current_year:
        st.info(f"💡 **Note:** Project started {current_year - start_year} years ago. RNAV calculation excludes historical cash flows and only considers future value from {current_year} onwards.")

    st.header("Cash Flow Chart")
    # Filter out the "Total" row and create chart with years on x-axis
    chart_data = df_rnav[~df_rnav["Year"].isin(["Total RNAV"])].copy()
    
    if len(chart_data) > 0:
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
        
        # Add vertical line at current year
        fig.add_vline(
            x=current_year, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Current Year ({current_year})",
            annotation_position="top"
        )
        
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
    else:
        st.warning("No data available for chart display.")

# Ensure main function is called when the script runs
if __name__ == "__main__":
    main()
else:
    # For Streamlit, also call main when imported as a module
    main()
