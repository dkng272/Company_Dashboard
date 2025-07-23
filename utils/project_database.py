import pandas as pd
import os
import datetime

def save_project_data(project_data, project_name, rnav_value=None):
    """Save project data to the CSV database"""
    try:
        csv_path = os.path.join("data", "real_estate_projects.csv")
        
        # Load existing data
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Create new dataframe with required columns
            columns = ['company_ticker', 'company_name', 'project_name', 'total_units', 
                      'average_selling_price', 'net_sellable_area', 'gross_floor_area', 
                      'land_area', 'construction_cost_per_sqm', 'land_cost_per_sqm',
                      'construction_start_year', 'sale_start_year', 'land_payment_year',
                      'construction_years', 'sales_years', 'revenue_booking_start_year', 
                      'project_completion_year', 'sga_percentage', 'wacc_rate', 'rnav_value',
                      'last_updated']
            df = pd.DataFrame(columns=columns)
        
        # Prepare new row data
        new_row = {
            'company_ticker': project_data.get('company_ticker', 'MANUAL'),
            'company_name': project_data.get('company_name', 'Manual Entry'),
            'project_name': project_name,
            'total_units': project_data['total_units'],
            'average_selling_price': project_data['average_selling_price'],
            'net_sellable_area': project_data['total_units'] * project_data['average_unit_size'],
            'gross_floor_area': project_data['gross_floor_area'],
            'land_area': project_data['land_area'],
            'construction_cost_per_sqm': project_data['construction_cost_per_sqm'],
            'land_cost_per_sqm': project_data['land_cost_per_sqm'],
            'construction_start_year': project_data['construction_start_year'],
            'sale_start_year': project_data['sale_start_year'],
            'land_payment_year': project_data['land_payment_year'],
            'construction_years': project_data['construction_years'],
            'sales_years': project_data['sales_years'],
            'revenue_booking_start_year': project_data['revenue_booking_start_year'],
            'project_completion_year': project_data['project_completion_year'],
            'sga_percentage': project_data['sga_percentage'],
            'wacc_rate': project_data['wacc_rate'],
            'rnav_value': rnav_value,
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Check if project already exists
        existing_index = df[(df['project_name'] == project_name) & 
                           (df['company_ticker'] == new_row['company_ticker'])].index
        
        if len(existing_index) > 0:
            # Update existing project
            for col, value in new_row.items():
                df.loc[existing_index[0], col] = value
            action = "updated"
        else:
            # Add new project
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            action = "saved"
        
        # Save to CSV
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return {"success": True, "message": f"Project {action} successfully!", "action": action}
        
    except Exception as e:
        return {"success": False, "message": f"Error saving project: {str(e)}"}

def load_project_data(project_name):
    """Load project data from the CSV database"""
    try:
        csv_path = os.path.join("data", "real_estate_projects.csv")
        df = pd.read_csv(csv_path)
        
        # Find project
        project_rows = df[df['project_name'] == project_name]
        
        if project_rows.empty:
            return {"success": False, "message": f"Project '{project_name}' not found in database"}
        
        if len(project_rows) > 1:
            # Multiple companies have projects with same name - show warning if possible
            print(f"Warning: Multiple projects found with name '{project_name}'. Using the first one.")
        
        project_data = project_rows.iloc[0].to_dict()
        
        # Calculate average unit size from NSA and total units
        if project_data['total_units'] > 0:
            project_data['average_unit_size'] = project_data['net_sellable_area'] / project_data['total_units']
        else:
            project_data['average_unit_size'] = 80  # default
        
        return {"success": True, "data": project_data, "message": "Project loaded successfully!"}
        
    except FileNotFoundError:
        return {"success": False, "message": "Project database not found"}
    except Exception as e:
        return {"success": False, "message": f"Error loading project: {str(e)}"}

def get_all_project_names():
    """Get list of all project names from database"""
    try:
        csv_path = os.path.join("data", "real_estate_projects.csv")
        df = pd.read_csv(csv_path)
        return df['project_name'].unique().tolist()
    except:
        return []

def load_projects_database():
    """Load the complete projects database"""
    try:
        csv_path = os.path.join("data", "real_estate_projects.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        else:
            print("Warning: No project database found. Please ensure 'data/real_estate_projects.csv' exists.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading project database: {str(e)}")
        return pd.DataFrame()

def get_companies_from_database():
    """Get list of unique companies from database"""
    df = load_projects_database()
    if df.empty:
        return []
    companies = df[['company_ticker', 'company_name']].drop_duplicates()
    return [f"{row['company_ticker']} - {row['company_name']}" for _, row in companies.iterrows()]

def get_projects_for_company(company_ticker):
    """Get projects for a specific company ticker"""
    df = load_projects_database()
    if df.empty:
        return []
    company_projects = df[df['company_ticker'] == company_ticker]
    return company_projects['project_name'].tolist()

def get_project_data_from_database(company_ticker, project_name):
    """Get specific project data from database"""
    df = load_projects_database()
    if df.empty:
        return None
    
    project_data = df[(df['company_ticker'] == company_ticker) & 
                     (df['project_name'] == project_name)]
    
    if project_data.empty:
        return None
    
    data = project_data.iloc[0].to_dict()
    # Calculate average unit size from NSA and total units
    if data['total_units'] > 0:
        data['average_unit_size'] = data['net_sellable_area'] / data['total_units']
    else:
        data['average_unit_size'] = 80  # default
    
    return data
