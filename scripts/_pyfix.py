with open(r"scripts\run_m1_causal_masked_analysis.py", "r", encoding="utf-8") as f:
    content = f.read()
# Remove all \" inside f-string expressions
import re
# Replace backslash-doublequote globally since they appear only in f-string dict accesses
content = content.replace('\\"', '"')
with open(r"scripts\run_m1_causal_masked_analysis.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed")
