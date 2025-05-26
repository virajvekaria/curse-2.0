def divide(dividend: int, divisor: int) -> float:
    """
    Divides two numbers and returns the result.
    
    Parameters:
    dividend (int): The number being divided.
    divisor (int): The number by which we are dividing.
    
    Returns:
    float: The result of the division.
    
    Raises:
    ValueError: If the divisor is 0.
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero")
    return dividend / divisor