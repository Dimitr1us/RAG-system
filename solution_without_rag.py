def find_max(numbers):
    if not numbers:
        return None
    
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
            
    return max_value