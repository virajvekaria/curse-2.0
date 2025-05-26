import math

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide two numbers with error handling for division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

if __name__ == "__main__":
    print("Welcome to the Simple Calculator!")
    while True:
        try:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
            operation = input("Choose operation (add, subtract, multiply, divide): ").lower()
            
            if operation == "add":
                result = add(num1, num2)
            elif operation == "subtract":
                result = subtract(num1, num2)
            elif operation == "multiply":
                result = multiply(num1, num2)
            elif operation == "divide":
                result = divide(num1, num2)
            else:
                print("Invalid operation. Please choose from add, subtract, multiply, or divide.")
                continue
            
            print(f"Result: {result}")
        except ValueError as e:
            print(e)
        
        another_calculation = input("Do you want to perform another calculation? (yes/no): ").lower()
        if another_calculation != "yes":
            break