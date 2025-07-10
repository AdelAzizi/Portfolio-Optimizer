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
        # This print statement is for debugging and can be removed in production tests.
        # print(f"Could not convert date: {date_input}, Error: {e}")
        # Return a 'Not a Time' value if conversion fails
        return pd.NaT

# --- Testing the function ---
test_dates = [14020510, '1401-11-22', 13991229, '1400/03/01', 'invalid-date']
gregorian_results = [robust_jalali_to_gregorian(d) for d in test_dates]

# --- Assertions to validate the function's behavior ---

# 1. Check successful conversions
assert gregorian_results[0] == pd.Timestamp('2023-08-01'), "Test Case 1 Failed"
assert gregorian_results[1] == pd.Timestamp('2023-02-11'), "Test Case 2 Failed"
assert gregorian_results[2] == pd.Timestamp('2021-03-19'), "Test Case 3 Failed"
assert gregorian_results[3] == pd.Timestamp('2021-05-22'), "Test Case 4 Failed"

# 2. Check handling of invalid data
assert pd.isna(gregorian_results[4]), "Test Case 5 (Invalid Date) Failed"

# 3. Verify the type of a valid output
assert isinstance(gregorian_results[0], pd.Timestamp), "Type check failed"

print("✅ All date conversion tests passed successfully.")
