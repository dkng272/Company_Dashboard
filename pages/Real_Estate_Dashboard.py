import streamlit as st
import pandas as pd
import os
from urllib.parse import urlencode

# Page configuration
st.set_page_config(
    page_title="Real Estate Dashboard",
    page_icon="🏢",
    layout="wide"
)

def load_projects_data():
    """Load real estate projects from CSV database"""
    try:
        csv_path = os.path.join("data", "real_estate_projects.csv")
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        st.error("❌ Real estate projects database not found. Please ensure 'data/real_estate_projects.csv' exists.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading projects data: {str(e)}")
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

def main():
    st.title("🏢 Real Estate Company Dashboard")
    
    # Load projects data
    df_projects = load_projects_data()
    
    if df_projects.empty:
        st.warning("No project data available. Please check the database file.")
        return
    
    # Sidebar for company selection
    st.sidebar.header("🔍 Company Selection")
    
    # Get unique companies
    companies = df_projects[['company_ticker', 'company_name']].drop_duplicates()
    company_options = [f"{row['company_ticker']} - {row['company_name']}" for _, row in companies.iterrows()]
    
    selected_company = st.sidebar.selectbox(
        "Select Company:",
        options=["Select a company..."] + company_options,
        index=0
    )
    
    if selected_company == "Select a company...":
        # Show overview of all companies
        st.header("📊 Company Overview")
        
        # Calculate summary statistics by company
        company_summary = df_projects.groupby(['company_ticker', 'company_name']).agg({
            'project_name': 'count',
            'total_units': 'sum',
            'net_sellable_area': 'sum',
            'average_selling_price': 'mean'
        }).reset_index()
        
        company_summary.columns = ['Ticker', 'Company Name', 'Total Projects', 'Total Units', 'Total NSA (m²)', 'Avg Price (VND/m²)']
        
        # Format the summary table
        for idx, row in company_summary.iterrows():
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Ticker", row['Ticker'])
            with col2:
                st.metric("Projects", f"{int(row['Total Projects'])}")
            with col3:
                st.metric("Total Units", format_units_display(row['Total Units']))
            with col4:
                st.metric("Total NSA", format_area_display(row['Total NSA (m²)']))
            with col5:
                st.metric("Avg Price", format_vnd_display(row['Avg Price (VND/m²)']))
            
            st.markdown("---")
        
        st.info("👆 Select a company from the sidebar to view detailed project information.")
        
    else:
        # Extract ticker from selection
        selected_ticker = selected_company.split(" - ")[0]
        
        # Filter projects for selected company
        company_projects = df_projects[df_projects['company_ticker'] == selected_ticker].copy()
        
        if company_projects.empty:
            st.warning(f"No projects found for {selected_company}")
            return
        
        # Display company header
        company_name = company_projects.iloc[0]['company_name']
        st.header(f"🏢 {selected_ticker} - {company_name}")
        
        # Company statistics
        total_projects = len(company_projects)
        total_units = company_projects['total_units'].sum()
        total_nsa = company_projects['net_sellable_area'].sum()
        avg_price = company_projects['average_selling_price'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Projects", total_projects)
        with col2:
            st.metric("Total Units", format_units_display(total_units))
        with col3:
            st.metric("Total NSA", format_area_display(total_nsa))
        with col4:
            st.metric("Avg Selling Price", format_vnd_display(avg_price))
        
        st.markdown("---")
        
        # Projects table
        st.subheader("📋 Project Portfolio")
        
        # Create display table with formatted values
        display_projects = company_projects.copy()
        
        # Format columns for display
        display_projects['Units'] = display_projects['total_units'].apply(format_units_display)
        display_projects['ASP (VND/m²)'] = display_projects['average_selling_price'].apply(format_vnd_display)
        display_projects['NSA'] = display_projects['net_sellable_area'].apply(format_area_display)
        display_projects['GFA'] = display_projects['gross_floor_area'].apply(format_area_display)
        display_projects['Land Area'] = display_projects['land_area'].apply(format_area_display)
        display_projects['Construction Cost'] = display_projects['construction_cost_per_sqm'].apply(format_vnd_display)
        display_projects['Land Cost'] = display_projects['land_cost_per_sqm'].apply(format_vnd_display)
        
        # Select columns for display
        display_cols = [
            'project_name', 'Units', 'ASP (VND/m²)', 'NSA', 'GFA', 'Land Area',
            'construction_start_year', 'sale_start_year', 'construction_years', 
            'sales_years', 'revenue_booking_start_year', 'project_completion_year'
        ]
        
        # Rename columns for better display
        column_names = {
            'project_name': 'Project Name',
            'construction_start_year': 'Constr. Start',
            'sale_start_year': 'Sale Start',
            'construction_years': 'Constr. Years',
            'sales_years': 'Sales Years',
            'revenue_booking_start_year': 'Rev. Booking',
            'project_completion_year': 'Completion'
        }
        
        display_table = display_projects[display_cols].rename(columns=column_names)
        
        # Use data editor for interactive table
        selected_indices = st.data_editor(
            display_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Project Name": st.column_config.TextColumn("Project Name", width="medium"),
                "Units": st.column_config.TextColumn("Units", width="small"),
                "ASP (VND/m²)": st.column_config.TextColumn("ASP", width="small"),
                "NSA": st.column_config.TextColumn("NSA", width="small"),
                "GFA": st.column_config.TextColumn("GFA", width="small"),
                "Land Area": st.column_config.TextColumn("Land", width="small"),
                "Constr. Start": st.column_config.NumberColumn("Start", width="small"),
                "Sale Start": st.column_config.NumberColumn("Sale", width="small"),
                "Constr. Years": st.column_config.NumberColumn("Years", width="small"),
                "Sales Years": st.column_config.NumberColumn("Years", width="small"),
                "Rev. Booking": st.column_config.NumberColumn("Booking", width="small"),
                "Completion": st.column_config.NumberColumn("Complete", width="small")
            },
            disabled=True
        )
        
        st.markdown("---")
        
        # Project selection for RNAV calculator
        st.subheader("🧮 Open RNAV Calculator")
        
        project_names = company_projects['project_name'].tolist()
        selected_project = st.selectbox(
            "Select a project to analyze:",
            options=["Select a project..."] + project_names,
            index=0
        )
        
        if selected_project != "Select a project...":
            # Get project data
            project_data = company_projects[company_projects['project_name'] == selected_project].iloc[0]
            
            # Display project summary
            st.info(f"📊 **Selected Project:** {selected_project}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Units:** {format_units_display(project_data['total_units'])}")
                st.write(f"**NSA:** {format_area_display(project_data['net_sellable_area'])}")
                st.write(f"**ASP:** {format_vnd_display(project_data['average_selling_price'])}")
            with col2:
                st.write(f"**GFA:** {format_area_display(project_data['gross_floor_area'])}")
                st.write(f"**Land Area:** {format_area_display(project_data['land_area'])}")
                st.write(f"**Construction Cost:** {format_vnd_display(project_data['construction_cost_per_sqm'])}")
            with col3:
                st.write(f"**Land Cost:** {format_vnd_display(project_data['land_cost_per_sqm'])}")
                st.write(f"**Construction:** {int(project_data['construction_start_year'])} - {int(project_data['construction_start_year'] + project_data['construction_years'] - 1)}")
                st.write(f"**Sales:** {int(project_data['sale_start_year'])} - {int(project_data['sale_start_year'] + project_data['sales_years'] - 1)}")
            
            # Button to open RNAV calculator
            if st.button(f"🧮 Open RNAV Calculator for {selected_project}", type="primary"):
                # Store project data in session state
                st.session_state['preload_project_data'] = project_data.to_dict()
                st.session_state['preload_project_name'] = selected_project
                
                # Show success message and navigation instruction
                st.success(f"✅ Project data loaded! Navigate to 'Real Estate RNAV Copy' page to see the pre-loaded calculator.")
                st.info("💡 **Next Step:** Use the sidebar navigation to go to 'Real Estate RNAV Copy' page.")

if __name__ == "__main__":
    main()
