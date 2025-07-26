import streamlit as st
import pandas as pd
import os
from pymongo import MongoClient
from datetime import datetime
import certifi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Real Estate Dashboard (MongoDB)",
    page_icon="🏢",
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
            st.warning(f"No companies found in MongoDB database '{db_name}', collection '{collection_name}'.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(companies_list)
        
        # Remove MongoDB ObjectId if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
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
            st.warning(f"No projects found in MongoDB database '{db_name}', collection '{collection_name}'.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_projects = pd.DataFrame(projects_list)
        
        # Remove MongoDB ObjectId if present
        if '_id' in df_projects.columns:
            df_projects = df_projects.drop('_id', axis=1)
        
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
            
            return df_merged
        else:
            # Handle date conversion for last_updated if it exists
            if 'last_updated' in df_projects.columns:
                df_projects['last_updated'] = pd.to_datetime(df_projects['last_updated'], errors='coerce')
            return df_projects
        
    except Exception as e:
        st.error(f"❌ Error loading projects data from MongoDB: {str(e)}")
        return pd.DataFrame()

def format_vnd_display(value):
    """Format VND values for display"""
    if pd.isna(value) or value == 0:
        return "N/A"
    
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:,.1f}B VND"
    elif value >= 1_000_000:
        return f"{value/1_000_000:,.0f}M VND"
    else:
        return f"{value:,.0f} VND"

def format_area_display(value):
    """Format area values for display"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"{value:,.0f} m²"

def format_units_display(value):
    """Format unit count for display"""
    if pd.isna(value) or value == 0:
        return "N/A"
    return f"{int(value):,} units"

def update_project_rnav(project_id, rnav_value):
    """Update RNAV value for a project in MongoDB"""
    try:
        client = init_mongodb_connection()
        if client is None:
            return False
        
        # Get database and collection names - using VietnamStocks database
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client.get_database(db_name)
        collection = db.get_collection(collection_name)
        
        # Update the project with new RNAV value and timestamp
        result = collection.update_one(
            {'project_id': project_id},  # Adjust the filter field as needed
            {
                '$set': {
                    'rnav_value': rnav_value,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
        )
        
        return result.modified_count > 0
        
    except Exception as e:
        st.error(f"❌ Error updating RNAV in MongoDB: {str(e)}")
        return False

def main():
    st.title("🏢 Real Estate Company Dashboard (MongoDB)")
    
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
            st.stop()
    
    # Load projects data (now includes company info)
    df_projects = load_projects_data()
    
    if df_projects.empty:
        st.warning("No project data available. Please check the MongoDB database.")
        return
    
    # Debug: Show available columns
    with st.sidebar:
        st.subheader("📋 Available Data")
        st.write(f"Total Projects: {len(df_projects)}")
        if 'company_ticker' in df_projects.columns:
            unique_companies = df_projects['company_ticker'].nunique()
            st.write(f"Unique Companies: {unique_companies}")
    
    # Sidebar for company selection
    st.sidebar.header("🔍 Company Selection")
    
    # Get unique companies - handle both cases where company_name might or might not be available
    if 'company_name' in df_projects.columns:
        companies = df_projects[['company_ticker', 'company_name']].dropna().drop_duplicates()
        company_options = [f"{row['company_ticker']} - {row['company_name']}" for _, row in companies.iterrows()]
    else:
        companies = df_projects[['company_ticker']].dropna().drop_duplicates()
        company_options = [row['company_ticker'] for _, row in companies.iterrows()]
    
    selected_company = st.sidebar.selectbox(
        "Select Company:",
        options=["Select a company..."] + company_options,
        index=0
    )
    
    if selected_company == "Select a company...":
        # Show overview of all companies
        st.header("📊 Company Overview")
        
        # Calculate summary statistics by company including RNAV
        if 'company_name' in df_projects.columns:
            group_cols = ['company_ticker', 'company_name']
        else:
            group_cols = ['company_ticker']
        
        company_summary = df_projects.groupby(group_cols).agg({
            'project_name': 'count',
            'total_units': 'sum',
            'net_sellable_area': 'sum',
            'average_selling_price': 'mean',
            'rnav_value': 'sum'
        }).reset_index()
        
        if 'company_name' in df_projects.columns:
            company_summary.columns = ['Ticker', 'Company Name', 'Total Projects', 'Total Units', 'Total NSA (m²)', 'Avg Price (VND/m²)', 'Total RNAV']
        else:
            company_summary.columns = ['Ticker', 'Total Projects', 'Total Units', 'Total NSA (m²)', 'Avg Price (VND/m²)', 'Total RNAV']
        
        # Sort by Total RNAV descending
        company_summary = company_summary.sort_values('Total RNAV', ascending=False, na_position='last')
        
        # Format the summary table
        for idx, row in company_summary.iterrows():
            if 'company_name' in df_projects.columns:
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric("Ticker", row['Ticker'])
                with col2:
                    st.metric("Company", row['Company Name'][:15] + "..." if len(str(row['Company Name'])) > 15 else row['Company Name'])
                with col3:
                    st.metric("Projects", f"{int(row['Total Projects'])}")
                with col4:
                    st.metric("Total Units", format_units_display(row['Total Units']))
                with col5:
                    st.metric("Avg Price", format_vnd_display(row['Avg Price (VND/m²)']))
                with col6:
                    st.metric("Total RNAV", format_vnd_display(row['Total RNAV']))
            else:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Ticker", row['Ticker'])
                with col2:
                    st.metric("Projects", f"{int(row['Total Projects'])}")
                with col3:
                    st.metric("Total Units", format_units_display(row['Total Units']))
                with col4:
                    st.metric("Avg Price", format_vnd_display(row['Avg Price (VND/m²)']))
                with col5:
                    st.metric("Total RNAV", format_vnd_display(row['Total RNAV']))
            
            st.markdown("---")
        
        st.info("👆 Select a company from the sidebar to view detailed project information and individual RNAVs.")
        
    else:
        # Extract ticker from selection
        if " - " in selected_company:
            selected_ticker = selected_company.split(" - ")[0]
        else:
            selected_ticker = selected_company
        
        # Filter projects for selected company
        company_projects = df_projects[df_projects['company_ticker'] == selected_ticker].copy()
        
        if company_projects.empty:
            st.warning(f"No projects found for {selected_company}")
            return
        
        # Display company header
        if 'company_name' in company_projects.columns and pd.notna(company_projects.iloc[0]['company_name']):
            company_name = company_projects.iloc[0]['company_name']
            st.header(f"🏢 {selected_ticker} - {company_name}")
        else:
            st.header(f"🏢 {selected_ticker}")
        
        # Company statistics including total RNAV
        total_projects = len(company_projects)
        total_units = company_projects['total_units'].sum()
        total_nsa = company_projects['net_sellable_area'].sum()
        avg_price = company_projects['average_selling_price'].mean()
        total_rnav = company_projects['rnav_value'].sum()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Projects", total_projects)
        with col2:
            st.metric("Total Units", format_units_display(total_units))
        with col3:
            st.metric("Total NSA", format_area_display(total_nsa))
        with col4:
            st.metric("Avg Selling Price", format_vnd_display(avg_price))
        with col5:
            st.metric("🏆 Total Portfolio RNAV", format_vnd_display(total_rnav))
        
        st.markdown("---")
        
        # Projects table with RNAV
        st.subheader("📋 Project Portfolio with RNAV")
        
        # Sort projects by RNAV descending
        company_projects = company_projects.sort_values('rnav_value', ascending=False, na_position='last')
        
        # Create display table with formatted values
        display_projects = company_projects.copy()
        
        # Format columns for display
        display_projects['Units'] = display_projects['total_units'].apply(format_units_display)
        display_projects['ASP (VND/m²)'] = display_projects['average_selling_price'].apply(format_vnd_display)
        display_projects['NSA'] = display_projects['net_sellable_area'].apply(format_area_display)
        display_projects['RNAV'] = display_projects['rnav_value'].apply(format_vnd_display)
        display_projects['Last Updated'] = pd.to_datetime(display_projects['last_updated'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Select columns for display - reordered to show RNAV prominently
        display_cols = [
            'project_name', 'location', 'RNAV', 'Units', 'ASP (VND/m²)', 'NSA',
            'construction_start_year', 'project_completion_year', 'Last Updated'
        ]
        
        # Rename columns for better display
        column_names = {
            'project_name': 'Project Name',
            'location': 'Location',
            'construction_start_year': 'Start Year',
            'project_completion_year': 'Completion Year'
        }
        
        display_table = display_projects[display_cols].rename(columns=column_names)
        
        # Use data editor for interactive table
        st.data_editor(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Project Name": st.column_config.TextColumn("Project Name", width="large"),
                "RNAV": st.column_config.TextColumn("🏆 RNAV", width="medium"),
                "Units": st.column_config.TextColumn("Units", width="small"),
                "ASP (VND/m²)": st.column_config.TextColumn("ASP", width="small"),
                "NSA": st.column_config.TextColumn("NSA", width="small"),
                "Start Year": st.column_config.NumberColumn("Start", width="small"),
                "Completion Year": st.column_config.NumberColumn("Complete", width="small"),
                "Last Updated": st.column_config.TextColumn("Updated", width="small")
            },
            disabled=True
        )
        
        
        st.markdown("---")
        
        

if __name__ == "__main__":
    main()
