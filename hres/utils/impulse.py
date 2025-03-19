def impulse(total_impulse):
    """
    Identifies impulse class of motor based on total delivered impulse
    
    Parameters:
    -----------
    total_impulse : float
        Total impulse in Newton-seconds
        
    Returns:
    --------
    tuple
        (motor_class, percent) - Motor class letter and percentage within class
    """
    impulse_classes = [
        (0, 1.25, 'A'),
        (1.25, 5, 'B'),
        (5, 10, 'C'),
        (10, 20, 'D'),
        (20, 40, 'E'),
        (40, 80, 'F'),
        (80, 160, 'G'),
        (160, 320, 'H'),
        (320, 640, 'I'),
        (640, 1280, 'J'),
        (1280, 2560, 'K'),
        (2560, 5120, 'L'),
        (5120, 10240, 'M'),
        (10240, 20480, 'N'),
        (20480, 40960, 'O'),
        (40960, 81920, 'P'),
        (81920, 163840, 'Q'),
        (163840, 327680, 'R'),
        (327680, 655360, 'S'),
        (655360, 1310720, 'T')
    ]
    
    for min_impulse, max_impulse, motor_class in impulse_classes:
        if min_impulse < total_impulse <= max_impulse:
            percent = 100.0 * (total_impulse - min_impulse) / (max_impulse - min_impulse)
            return motor_class, percent
            
    # If outside range
    return None, None
