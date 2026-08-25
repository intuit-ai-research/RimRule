RULE_PROMPT = """You are a senior AI systems instructor helping a tool-using language model improve.
Compare the failed trace with the ground-truth trace. Identify the earliest root-cause reasoning error, not downstream consequences.
Check decomposition, tool selection, and tool arguments. Generate exactly one atomic, reusable rule.
The rule must not contain query-specific entities or concrete tool names and must use a clean if-then form.
Return JSON only: {{\"rule\": \"If ... then ...\", \"error_type\": \"decomposition|tool_selection|arguments\"}}.

QUERY:
{query}

TOOLS:
{tools}

FAILED TRACE:
{failed}

GROUND-TRUTH TRACE:
{reference}

PREVIOUS ACCEPTED RULES FOR THIS FAILURE:
{previous}
"""

VOCAB_PROMPT = """Induce compact closed vocabularies for canonicalizing these rules.
Return JSON with exactly five arrays: domain, qualifier, action, strength, tool_category.
Prefer reusable uppercase snake-case labels and minimize total vocabulary size without losing distinctions.
RULES:\n{rules}"""

TRANSLATE_PROMPT = """Translate the rule into the fixed vocabulary. Use only listed values and return JSON with exactly five arrays:
domain, qualifier, action, strength, tool_category. Empty arrays are allowed.
VOCABULARY:\n{vocabulary}\nRULE:\n{rule}"""

QUERY_SYMBOL_PROMPT = """Translate this ToolHop query and its available tools into the fixed vocabulary.
Return JSON with arrays domain, qualifier, action, strength, tool_category. Use only listed values.
VOCABULARY:\n{vocabulary}\nQUERY:\n{query}\nTOOLS:\n{tools}"""
