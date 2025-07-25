#%%

import streamlit as st
import os
import pandas as pd
import certifi
from pymongo import MongoClient
import datetime
from dotenv import load_dotenv


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

# %%
