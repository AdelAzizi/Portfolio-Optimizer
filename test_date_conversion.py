import jdatetime
import pandas as pd

def robust_jalali_to_gregorian(date_input) -> pd.Timestamp:
    """Converts various Jalali date formats (int or str) to Gregorian datetime."""
    try:
        # Convert integer to string and normalize separators
        date_str = str(date_input).replace('/', '-')
        
        # Check if the date string contains separators
        if '-' in date_str:
            y, m, d = map(int, date_str.split('-'))
        else: # Assumes a dense format like 14020510
            y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:])
        
        # Convert to Gregorian date and then to pandas Timestamp
        gregorian_date = jdatetime.date(y, m, d).togregorian()
        return pd.to_datetime(gregorian_date)
        
    except (ValueError, TypeError) as e:
        print(f"Could not convert date: {date_input}, Error: {e}")
        # Return a 'Not a Time' value if conversion fails
        return pd.NaT

# --- Testing the function ---
test_dates = [14020510, '1401-11-22', 13991229, '1400/03/01', 'invalid-date']
gregorian_results = [robust_jalali_to_gregorian(d) for d in test_dates]

print("Original Jalali Dates:", test_dates)
print("Converted Gregorian Dates:", gregorian_results)

# Verify the type of the output for a valid date
if gregorian_results and pd.notna(gregorian_results[0]):
    print("Type of first result:", type(gregorian_results[0]))
