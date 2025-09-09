correct_input = False
while not correct_input :
    try:
        height = float(input("Enter your height: "))
        weight = float(input("Enter your weight: "))
    except ValueError:
        print("Enter Valid Data")
    else:
        correct_input = True
