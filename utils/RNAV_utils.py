#%%
from dotenv import load_dotenv
import pandas as pd
import requests
import re

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