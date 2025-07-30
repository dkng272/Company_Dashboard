#%%
import pymongo
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

load_dotenv()

#%% MongoDB Connection
@st.cache_resource
def get_mongo_client():
    """Get MongoDB client with connection pooling"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        mongo_uri = st.secrets["MONGODB_URI"]
    except:
        # Fall back to .env file (for local development)
        mongo_uri = os.getenv('MONGODB_URI')
    
    if not mongo_uri:
        st.error("MongoDB URI not found. Please set MONGODB_URI in .env file or Streamlit secrets.")
        st.stop()
    
    return MongoClient(mongo_uri)

@st.cache_resource
def get_databases():
    """Get database references"""
    client = get_mongo_client()
    return {
        'vnstock_data': client['vnstock_data'],
        'classification': client['classification']
    }

#%% Classification database query functions
def query_collection(collection_name, query={}, projection=None, limit=None, db_name='classification'):
    """Query MongoDB collection and return DataFrame"""
    dbs = get_databases()
    db = dbs[db_name]
    collection = db[collection_name]
    
    cursor = collection.find(query, projection)
    if limit:
        cursor = cursor.limit(limit)
    
    df = pd.DataFrame(list(cursor))
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    return df

@st.cache_data
def get_tickers_names():
    """Get ticker names mapping from classification database"""
    return query_collection('tickers_names')

@st.cache_data
def get_stock_list():
    """Get stock list from classification database"""
    return query_collection('stock_list')

@st.cache_data
def get_bank_keycodes():
    """Get bank keycodes mapping from classification database"""
    return query_collection('bank_keycodes')

@st.cache_data
def get_bank_classification():
    """Get bank classification from classification database"""
    return query_collection('bank_classification')