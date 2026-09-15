#!/usr/bin/env python3
"""Test the submit endpoint"""
import sys
sys.path.insert(0, '/home/ubuntu/meowify-v2')

import traceback
try:
    from server import submit
    print("✓ server.py imports successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# Check the function signature
import inspect
sig = inspect.signature(submit)
print(f"\nsubmit() parameters:")
for name, param in sig.parameters.items():
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    print(f"  {name}: {default}")
