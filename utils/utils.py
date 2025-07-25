from pathlib import Path

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parents[1]  # assuming utils/ is 1 level below root

def get_data_path(filename: str) -> Path:
    """Returns the full path to a file in the /data directory."""
    return get_project_root() / "data" / filename




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
