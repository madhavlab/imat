import os
import sys

def check_path(path):
    """Check if a path exists and what it contains."""
    print(f"Checking path: {path}")
    if os.path.exists(path):
        print(f"✓ Path exists: {path}")
        
        if os.path.isdir(path):
            print(f"✓ Path is a directory")
            # List contents
            contents = os.listdir(path)
            print(f"Contents ({len(contents)} items):")
            for item in contents:
                full_item_path = os.path.join(path, item)
                if os.path.isdir(full_item_path):
                    print(f"  DIR: {item}")
                else:
                    print(f"  FILE: {item}")
        else:
            print(f"✗ Path is not a directory")
    else:
        print(f"✗ Path does not exist")

# Your specific path
specific_path = "/home/kavyar/Documents/PhD/Final_Works/JOSS/imat/test_data/audio"
check_path(specific_path)

# Check parent directories to make sure they exist
parent = os.path.dirname(specific_path)  # test_data
check_path(parent)

parent_of_parent = os.path.dirname(parent)  # imat
check_path(parent_of_parent)

# Check current working directory
cwd = os.getcwd()
print(f"\nCurrent working directory: {cwd}")
check_path(cwd)

# See where the script thinks the project root is
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.dirname(script_dir)

print(f"\nScript path: {script_path}")
print(f"Script directory: {script_dir}")
print(f"Presumed project root: {project_root}")

# Check if we can access the path relative to the current working directory
relative_path = os.path.join(cwd, "test_data", "audio")
check_path(relative_path)

if __name__ == "__main__":
    print("Path debugging complete.")