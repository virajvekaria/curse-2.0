import json

def divide(dividend, divisor):
    """Divide two numbers with error handling.
    
    Args:
        dividend (int): The number being divided.
        divisor (int): The number by which the dividend is being divided.
    
    Returns:
        int: The result of the division.
    
    Raises:
        ValueError: If the divisor is 0.
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")
    return dividend // divisor

def main():
    # Example usage
    print(divide(10, 2))  # Output: 5
    print(divide(10, 0))  # Raises a ValueError

if __name__ == "__main__":
    main()