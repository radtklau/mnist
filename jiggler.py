import ctypes
import time
import random

# Define the structure for mouse input
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

# Define the structure for input
class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("mi", MOUSEINPUT)]

# Constants for mouse events
MOUSEEVENTF_MOVE = 0x0001

# Function to jiggle the mouse
def move_mouse_jiggle():
    while True:
        # Generate random movement within a threshold
        move_x = random.randint(-5, 5)
        move_y = random.randint(-5, 5)

        # Create a MOUSEINPUT structure
        mouse_input = MOUSEINPUT(move_x, move_y, 0, MOUSEEVENTF_MOVE, 0, None)

        # Create an INPUT structure
        input_structure = INPUT(0, mouse_input)

        # Send the mouse input
        ctypes.windll.user32.SendInput(1, ctypes.byref(input_structure), ctypes.sizeof(INPUT))

        # Pause for a short time (adjust as needed)
        time.sleep(5)

if __name__ == "__main__":
    try:
        move_mouse_jiggle()
    except KeyboardInterrupt:
        print("\nMouse Jiggler stopped.")
