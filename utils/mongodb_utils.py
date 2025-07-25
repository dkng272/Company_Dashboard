"""
MongoDB utilities for RNAV Calculator
Provides database operations for project data storage and retrieval
"""

import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    st.error("PyMongo not installed. Run: pip install pymongo")

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_CONNECTION_STRING")
DATABASE_NAME = os.getenv("MONGODB_DATABASE")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION")

def get_mongodb_client():
    """Get MongoDB client connection"""
    if not PYMONGO_AVAILABLE:
        return None
    
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        st.error(f"MongoDB connection failed: {str(e)}")
        return None

def test_mongodb_connection():
    """Test MongoDB connection and return status"""
    if not PYMONGO_AVAILABLE:
        return {"success": False, "message": "PyMongo not installed"}
    
    try:
        client = get_mongodb_client()
        if client is None:
            return {"success": False, "message": "Failed to create client"}
        
        # Test connection
        client.admin.command('ping')
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Count documents
        count = collection.count_documents({})
        client.close()
        
        return {
            "success": True, 
            "message": f"Connected to {DATABASE_NAME}.{COLLECTION_NAME} ({count} projects)"
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

def get_companies_from_database():
    """Get list of unique companies from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return []
    
    try:
        client = get_mongodb_client()
        if client is None:
            return []
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'Companies'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Get all companies
        companies_cursor = collection.find({})
        companies_list = list(companies_cursor)
        client.close()
        
        # Format as "TICKER - Company Name"
        formatted_companies = []
        for company in companies_list:
            ticker = company.get("ticker", "")
            name = company.get("company_name", "")
            if ticker and name:
                formatted_companies.append(f"{ticker} - {name}")
        
        return sorted(formatted_companies)
        
    except Exception as e:
        st.error(f"Error getting companies: {str(e)}")
        return []

def get_projects_for_company(company_ticker):
    """Get list of projects for a specific company from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return []
    
    try:
        client = get_mongodb_client()
        if client is None:
            return []
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Find projects for the company
        projects = collection.find(
            {"company_ticker": company_ticker},
            {"project_name": 1, "_id": 0}
        ).sort("project_name", 1)
        
        project_names = [project["project_name"] for project in projects]
        client.close()
        
        return project_names
        
    except Exception as e:
        st.error(f"Error getting projects for company: {str(e)}")
        return []

def get_project_data_from_database(company_ticker, project_name):
    """Get specific project data from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return None
    
    try:
        client = get_mongodb_client()
        if client is None:
            return None
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Find specific project
        project = collection.find_one({
            "company_ticker": company_ticker,
            "project_name": project_name
        })
        
        client.close()
        
        if project:
            # Remove MongoDB ObjectId
            project.pop('_id', None)
            return project
        else:
            return None
            
    except Exception as e:
        st.error(f"Error getting project data: {str(e)}")
        return None

def save_project_data(project_data, project_name, rnav_value=None):
    """Save project data to MongoDB"""
    if not PYMONGO_AVAILABLE:
        return {"success": False, "message": "PyMongo not available"}
    
    try:
        client = get_mongodb_client()
        if client is None:
            return {"success": False, "message": "Failed to connect to MongoDB"}
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Prepare document
        document = {
            "project_name": project_name,
            "company_ticker": project_data.get('company_ticker', 'MANUAL'),
            "company_name": project_data.get('company_name', 'Manual Entry'),
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
            "last_updated": datetime.now(),
            "created_date": datetime.now()
        }
        
        # Check if project exists
        existing = collection.find_one({
            "project_name": project_name,
            "company_ticker": document["company_ticker"]
        })
        
        if existing:
            # Update existing document but preserve created_date
            document["created_date"] = existing.get("created_date", datetime.now())
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
        
        client.close()
        return {"success": True, "message": message, "action": action}
        
    except Exception as e:
        return {"success": False, "message": f"Error saving to MongoDB: {str(e)}"}

def load_project_data(project_name):
    """Load project data by name from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return {"success": False, "message": "PyMongo not available"}
    
    try:
        client = get_mongodb_client()
        if client is None:
            return {"success": False, "message": "Failed to connect to MongoDB"}
        
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        # Find project by name
        project = collection.find_one({"project_name": project_name})
        client.close()
        
        if project:
            # Remove MongoDB ObjectId for JSON serialization
            project.pop('_id', None)
            return {"success": True, "data": project}
        else:
            return {"success": False, "message": f"Project '{project_name}' not found"}
            
    except Exception as e:
        return {"success": False, "message": f"Error loading from MongoDB: {str(e)}"}

def get_all_project_names():
    """Get list of all project names from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return []
    
    try:
        client = get_mongodb_client()
        if client is None:
            return []
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Get distinct project names
        project_names = collection.distinct("project_name")
        client.close()
        
        return sorted(project_names)
        
    except Exception as e:
        st.error(f"Error getting project names: {str(e)}")
        return []

def load_projects_database():
    """Load all projects as DataFrame from MongoDB"""
    if not PYMONGO_AVAILABLE:
        return pd.DataFrame()
    
    try:
        client = get_mongodb_client()
        if client is None:
            return pd.DataFrame()
        
        # Use same database and collection as Real_Estate_Dashboard_MongoDB
        db_name = 'VietnamStocks'
        collection_name = 'RealEstateProjects'
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Get all projects
        projects = list(collection.find({}, {"_id": 0}))  # Exclude _id field
        client.close()
        
        if projects:
            return pd.DataFrame(projects)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading projects database: {str(e)}")
        return pd.DataFrame()
