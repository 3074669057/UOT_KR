import re
path = r"scripts\run_m1_causal_masked_analysis.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix all f-string backslash/single-quote issues by pre-extracting variables
fixes = [
    ("len(ctx['eth_flows'])", "n_eth"),
    ("len(ctx['bnb_flows'])", "n_bnb"),
    ("len(ctx['truth'])", "n_truth"),
    ("ctx['mask_density']", "mask_d"),
    ("paper_ctx['mask_density']", "paper_mask_d"),
]
for old_expr, new_var in fixes:
    # Find the preceding line to insert assignment
    pattern = re.escape(old_expr)
    content = content.replace(old_expr, "{" + new_var + "}")

# Add variable assignments before the problematic prints
additions = [
    ("n_eth = len(ctx['eth_flows'])", "n_eth"),
    ("n_bnb = len(ctx['bnb_flows'])", "n_bnb"),
    ("n_truth = len(ctx['truth'])", "n_truth"),
    ("mask_d = ctx['mask_density']", "mask_d"),
    ("paper_mask_d = paper_ctx['mask_density']", "paper_mask_d"),
]

# Actually a simpler approach: use .format() instead of f-strings
# Replace all single-quoted f-strings with format calls
lines = content.split("\n")
new_lines = []
for line in lines:
    # Replace problematic f'...{...}...' patterns
    if "f'" in line and "ctx['" in line:
        # Replace with .format() call
        import_re = re.findall(r"ctx\['(\w+)'\]", line)
        for key in import_re:
            line = line.replace(f"ctx['{key}']", "{%s}" % key)
        # Convert f'...{var}...' to '...{var}...'.format(var=ctx['var'])
        line = line.replace("f'", "'").replace("{", "{").replace("}", "}")
        # Add .format call
        if "print('" in line and not line.rstrip().endswith(".format("):
            fmt_args = ", ".join(f"{k}=ctx['{k}']" for k in import_re)
            line = line.rstrip() + f".format({fmt_args})" if import_re else line
    new_lines.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(new_lines))
print("Fixed all f-string issues")
