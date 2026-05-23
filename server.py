# MCP server: connects to the MySQL database and exposes it to the client.
# Provides a schema resource (fetched once at startup) and three semantic tools
# (lookup_codes, list_distinct_values, execute_query) following the Claude Code
# pattern of specific tools for specific jobs.
from fastmcp import FastMCP
import mysql.connector
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

mcp = FastMCP("MySQL_Server")

db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "tfm_datanex",
    "port": 3306
}

# ----------------------------
# Domain descriptions for tables that need clarification.
# Add or edit entries here to guide the LLM toward the right table.
# ----------------------------
TABLE_DESCRIPTIONS = {
    "g_demographics": "Primary patient table. One row per patient. Use this to count patients or get demographic info. sex: 1=male/hombre, 2=female/mujer. natio_descr sample values: Espana, Europa, América, África, Marruecos, Pakistan, Andorra, Resto del mundo. birth_date is stored as TEXT in 'YYYY-MM-DD' format — use TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) to calculate age.",
    "g_administrations": "Drug administration records — one row per drug actually given to a patient. drug_descr is the full drug name with dosage form. atc_descr is the standardized drug name (e.g. 'Enoxaparina', 'Paracetamol', 'Meropenem'). Use atc_descr for filtering — do a lookup first: SELECT DISTINCT atc_descr FROM g_administrations WHERE atc_descr LIKE '%keyword%'. route_descr is the administration route (e.g. 'ORAL', 'INTRAVENOSA', 'SUBCUTANEA'). given = 'X' means the drug was actually administered; NULL means it was not given. quantity is the amount given, quantity_unit is the unit. episode_ref is bigint — can join with g_episodes, g_diagnostics on BOTH patient_ref AND episode_ref.",
    "g_health_issues": "Patient diagnoses and health conditions. Use this for any question about diseases, conditions, or diagnoses. Column snomed_descr contains the condition name in Spanish. snomed_ref is the numeric SNOMED code — use it for reliable filtering when known. Always SELECT and GROUP BY snomed_descr to include condition names in results. ou_med_ref is the medical unit that RECORDED the diagnosis — this is NOT necessarily the unit that treats the patient. Only filter ou_med_ref here when the question specifically asks 'diagnosed by' a specialty. For 'attended by' or 'seen by' a specialty, filter ou_med_ref on g_labs or g_episodes instead. episode_ref is stored as double — join with g_labs, g_procedures and g_episodes on patient_ref only (episode_ref type incompatible with those tables).",
    "g_labs": "Lab test results. Use this for any question about lab values or analytical results (e.g. PCR, glucose, hemoglobin). result_num is the numeric result. lab_descr is the test name in Spanish (e.g. 'Hemoglobina glicada' for HbA1c, 'Glucosa' for glucose). lab_sap_ref is the stable lab code — always do a lookup first (SELECT DISTINCT lab_sap_ref, lab_descr FROM g_labs WHERE lab_descr LIKE '%keyword%') and then filter by lab_sap_ref IN (...) in the main query. extrac_date is when the blood sample was drawn; result_date is when the result was reported — use extrac_date when comparing timing to procedures. ou_med_ref is the unit that ORDERED the lab — filter here when the question asks about patients 'attended by' or 'seen by' a specialty. Join with g_health_issues on patient_ref only (episode_ref types are incompatible). Join with g_procedures and g_episodes on BOTH patient_ref AND episode_ref to stay within the same episode.",
    "g_micro": "Microbiology results (cultures, microorganisms). Use this for questions about infections, cultures, or microorganisms — NOT for lab values like PCR. Key columns: mue_descr is the sample type (e.g. 'Hemocultiu' = blood culture / hemocultivo, 'Orina' = urine culture, 'Mostra respiratòria' = respiratory sample). micro_descr is the microorganism name (e.g. 'Escherichia coli', 'Pseudomonas aeruginosa'). positive = 'X' means the culture was positive (grew a microorganism); NULL means negative or pending. ALWAYS filter by mue_descr when the question specifies a sample type (hemocultivo → mue_descr = 'Hemocultiu'). Do a lookup first if unsure: SELECT DISTINCT mue_descr FROM g_micro. Join with g_episodes on BOTH patient_ref AND episode_ref.",
    "g_procedures": "Clinical procedures performed on patients. Use this for ANY question about procedures or interventions — prefer this over g_surgery. Column descr contains the procedure name in Spanish (plain ASCII, no accents). Column code contains procedure codes. ALWAYS look up codes first using BOTH the procedure type keyword AND the anatomy keyword: SELECT DISTINCT code, descr FROM g_procedures WHERE descr LIKE '%procedure_type%' AND descr LIKE '%anatomy%'. Then use ALL returned codes in code IN (...) in the main query. Join with g_labs and g_episodes on BOTH patient_ref AND episode_ref — NEVER join on patient_ref alone, as that cross-matches across episodes.",
    "g_surgery": "Surgical planning and scheduling data. Use only for questions specifically about surgical scheduling, waiting lists, or surgical teams — NOT for querying what procedures were performed (use g_procedures for that).",
    "g_episodes": "Hospital episodes/admissions. episode_type_ref='HOSP' means hospitalization (ingreso hospitalario) — ALWAYS filter episode_type_ref='HOSP' when the question asks about ingresos/hospitalizations. start_date is the admission date. Can join with g_procedures and g_labs on both patient_ref AND episode_ref.",
    "g_movements": "Patient movements between hospital units/locations. ou_med_ref and ou_med_descr map medical unit codes to names (e.g. ou_med_ref='CAR' → ou_med_descr='CARDIOLOGIA'). ou_loc_ref and ou_loc_descr identify the physical location/ward (e.g. ou_loc_ref='I064' → a specific ward). Use this table to find patients who were physically in a specific unit or ward. JOIN on patient_ref and episode_ref. To find a location code, query: SELECT DISTINCT ou_loc_ref, ou_loc_descr FROM g_movements WHERE ou_loc_descr LIKE '%keyword%'.",
    "g_diagnostics": "ICD-coded diagnoses (separate from g_health_issues which uses SNOMED). Column code contains ICD-9/ICD-10 codes (e.g. 'I63.9' for cerebral infarction). diag_descr is the diagnosis name. Use LIKE 'code%' to match all subcodes in a family (e.g. code LIKE 'I63%' for all cerebral infarction subtypes). Prefer g_diagnostics with ICD code ranges when the question mentions ICD codes or when broader diagnostic coverage is needed. episode_ref is bigint — can join with g_episodes, g_procedures, g_labs on both patient_ref AND episode_ref.",
    "g_antibiograms": "Antibiotic sensitivity results linked to microbiology cultures. micro_descr is the microorganism. antibiotic_descr is the antibiotic tested. sensitivity is the result: 'S' = sensitive, 'R' = resistant, 'I' = intermediate. result is the MIC value (may contain operators like '> 16', '<= 2'). Join with g_micro via patient_ref + antibiogram_ref to link sensitivity results to the culture sample type. Use this for questions about antibiotic resistance or sensitivity.",
    "g_prescriptions": "Drug prescriptions (what was ordered, not necessarily administered — for actual administration use g_administrations). atc_descr is the standardized drug name. drug_descr is the full drug description. start_drug_date and end_drug_date define the prescription period. dose is the prescribed dose, unit is the dose unit. route_descr is the administration route. episode_ref is bigint — can join with g_episodes on both patient_ref AND episode_ref.",
    "g_exitus": "Patient death records. One row per deceased patient. exitus_date is the date of death. Use this for mortality questions — join with g_demographics on patient_ref. To check if a patient died: EXISTS (SELECT 1 FROM g_exitus WHERE patient_ref = ...).",
    "g_diagnostic_related_groups": "DRG (Diagnosis Related Groups) data per episode. severity_descr indicates case severity: 'Minor', 'Moderate', 'Major', 'Extreme'. mortality_risk_descr indicates mortality risk: 'Minor', 'Moderate', 'Major', 'Extreme'. weight is the DRG weight (resource intensity). Use for severity analysis or case-mix questions. Join with g_episodes on both patient_ref AND episode_ref.",
    "g_encounters": "Outpatient visits and encounters. encounter_type distinguishes visit types. Join with g_episodes on patient_ref and episode_ref.",
}

# ----------------------------
# RESOURCE: Full live schema
# ----------------------------
@mcp.resource("schema://full")
def get_full_schema() -> str:
    """Return all tables with their columns, types, and descriptions, fetched live from the database."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]

        lines = []
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`;")
            columns = cursor.fetchall()
            cols_str = ", ".join(f"{col[0]} {col[1]}" for col in columns)
            desc = TABLE_DESCRIPTIONS.get(table, "")
            prefix = f"  # {desc}" if desc else ""
            lines.append(f"{table}({cols_str}){prefix}")

        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------------
# Internal helpers (not exposed as MCP tools)
# ----------------------------
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
# Heuristic: the canonical "code" column for each lookupable table.
_CODE_COLUMN_BY_TABLE = {
    "g_diagnostics": "code",
    "g_labs": "lab_sap_ref",
    "g_movements": "ou_med_ref",
    "g_health_issues": "snomed_ref",
    "g_administrations": "atc_descr",
    "g_prescriptions": "atc_descr",
    "g_procedures": "code",
    "g_micro": "micro_descr",
}


def _safe_identifier(name: str) -> bool:
    """Reject anything that isn't a plain SQL identifier (alphanumeric + underscore)."""
    return bool(_IDENTIFIER_RE.match(name))


def _run_select(sql: str) -> str:
    """Run a SELECT and return rows as JSON. Internal use only — assumes caller already validated."""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(sql)
        column_names = [desc[0] for desc in cursor.description]
        results = [{column_names[i]: row[i] for i in range(len(row))} for row in cursor.fetchall()]
        conn.close()
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Database Error: {str(e)}"


# ----------------------------
# Ontology loading — parses two OWL files once at startup.
#   ontology_input_v2.owl  → table-level classes (one per DB table)
#   ontology_input_dic.owl → dictionary classes for coded reference columns
# ----------------------------
_RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_OWL_NS = "{http://www.w3.org/2002/07/owl#}"
_RDFS_NS = "{http://www.w3.org/2000/01/rdf-schema#}"
_HERE = Path(__file__).parent


def _local_name(uri: str) -> str:
    """Strip the namespace prefix off a URI: '.../#GLabs' → 'GLabs'."""
    return uri.split("#")[-1] if "#" in uri else uri.rsplit("/", 1)[-1]


def _parse_ontology(file_path: Path) -> list[dict]:
    """Parse an OWL/RDF-XML file into a list of class records.
    Each record has: name, label, comment, parents (list of parent class local names).
    Returns [] if the file is missing or malformed.
    """
    if not file_path.exists():
        return []
    try:
        root = ET.parse(file_path).getroot()
    except ET.ParseError:
        return []
    classes = []
    for cls in root.findall(f"{_OWL_NS}Class"):
        about = cls.get(f"{_RDF_NS}about")
        if not about:
            continue
        name = _local_name(about)
        label_elem = cls.find(f"{_RDFS_NS}label")
        label = label_elem.text.strip() if label_elem is not None and label_elem.text else name
        comment_elem = cls.find(f"{_RDFS_NS}comment")
        comment = comment_elem.text.strip() if comment_elem is not None and comment_elem.text else ""
        parents = []
        for sub in cls.findall(f"{_RDFS_NS}subClassOf"):
            parent_ref = sub.get(f"{_RDF_NS}resource")
            if parent_ref:
                parents.append(_local_name(parent_ref))
                continue
            nested = sub.find(f"{_OWL_NS}Class")
            if nested is not None:
                nested_about = nested.get(f"{_RDF_NS}about")
                if nested_about:
                    parents.append(_local_name(nested_about))
        classes.append({"name": name, "label": label, "comment": comment, "parents": parents})
    return classes


_ONTOLOGY_TABLES = _parse_ontology(_HERE / "ontology_input_v2.owl")
_ONTOLOGY_DIC = _parse_ontology(_HERE / "ontology_input_dic.owl")


# ----------------------------
# TOOL 1: lookup_ontology — query the OWL ontology for schema-level concepts
# ----------------------------
@mcp.tool()
def lookup_ontology(keyword: str = "") -> str:
    """Search the OWL ontology for schema-level concepts that describe tables and their
    coded reference columns. Call this FIRST when you need to clarify what a Spanish-language
    column or coded reference field actually represents before looking up codes or values.

    The ontology has two layers, both loaded at startup:
      - tables:     one class per DB table (GDemographics, GLabs, GEpisodes, ...).
      - dictionary: classes for coded reference columns inside tables
                    (FreqRef → freq_ref in g_prescriptions, EncounterType → encounter_type
                    in g_encounters, MotType → mot_type in g_adm_disch, ...).

    Each result has: source ('tables' | 'dictionary'), name, label, comment (Spanish
    description), and parents (parent classes — for dictionary entries the parent is
    the owning table class).

    Typical workflow:
        1. lookup_ontology(keyword)              — understand what the concept means
        2. lookup_codes / list_distinct_values   — get the actual codes/values to filter on
        3. execute_query                          — final answer

    Args:
        keyword: Substring to match in the class name, label, or comment (case-insensitive).
                 Empty string returns every concept in both ontologies.

    Examples:
        lookup_ontology('freq')         → FreqRef (parent: GPrescriptions, freq_ref column)
        lookup_ontology('encounter')    → EncounterType (parent: GEncounters)
        lookup_ontology('demographics') → GDemographics table class
    """
    kw = keyword.lower().strip()
    results = []
    for source, classes in (("tables", _ONTOLOGY_TABLES), ("dictionary", _ONTOLOGY_DIC)):
        for c in classes:
            if (not kw
                    or kw in c["name"].lower()
                    or kw in c["label"].lower()
                    or kw in c["comment"].lower()):
                results.append({"source": source, **c})
    return json.dumps(results, ensure_ascii=False)


# ----------------------------
# TOOL 2: lookup_codes — find IDs/codes by description keyword
# ----------------------------
@mcp.tool()
def lookup_codes(table: str, descr_column: str, keyword: str, limit: int = 50) -> str:
    """Find codes/IDs in a code-bearing table by a keyword in the description column.

    Use this BEFORE filtering on any coded column when the user mentions a clinical concept by
    name (a diagnosis, a lab test, a medical specialty, a drug). Returns DISTINCT
    (code, description) pairs whose description contains the keyword (case-insensitive).

    Avoid execute_query for code discovery — call this tool instead.

    Args:
        table: Table to search. Common choices:
               'g_diagnostics' (ICD codes), 'g_labs' (lab tests),
               'g_movements' (medical-unit codes), 'g_health_issues' (SNOMED),
               'g_administrations'/'g_prescriptions' (drugs), 'g_procedures', 'g_micro'.
        descr_column: The description column name (e.g. 'diag_descr', 'lab_descr',
               'ou_med_descr', 'snomed_descr', 'atc_descr', 'descr', 'micro_descr').
        keyword: Substring to match. Use plain ASCII when possible (e.g. 'diabetes' not
               'diabéticos'); the search is case-insensitive.
        limit: Max rows to return (default 50, max 200).

    Examples:
        lookup_codes('g_diagnostics', 'diag_descr', 'diabetes')
            → ICD codes E10.21, E11.65, etc.
        lookup_codes('g_labs', 'lab_descr', 'HbA1c')
            → lab_sap_ref values LABHG1, LABHG2, LABHG3, LAB1368.
        lookup_codes('g_movements', 'ou_med_descr', 'cardio')
            → ou_med_ref CAR.
    """
    if not _safe_identifier(table) or not _safe_identifier(descr_column):
        return "Error: invalid table or column name."
    code_col = _CODE_COLUMN_BY_TABLE.get(table)
    if not code_col:
        return (f"Error: unknown lookup table '{table}'. Supported: "
                f"{', '.join(sorted(_CODE_COLUMN_BY_TABLE))}")
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 50, 200))
    safe_keyword = keyword.replace("'", "''")  # escape single quotes for the literal
    sql = (f"SELECT DISTINCT {code_col}, {descr_column} FROM {table} "
           f"WHERE LOWER({descr_column}) LIKE LOWER('%{safe_keyword}%') "
           f"LIMIT {safe_limit}")
    return _run_select(sql)


# ----------------------------
# TOOL 3: list_distinct_values — enumerate values of a varchar column
# ----------------------------
@mcp.tool()
def list_distinct_values(table: str, column: str, limit: int = 30) -> str:
    """List the distinct values stored in a text/varchar column.

    Use this BEFORE filtering on any varchar column whose actual values you don't know.
    Avoids guessing literals like 'España' vs 'Espana', 'Catalunya' vs 'Cataluña',
    'HOSP' vs 'hospitalization', etc.

    Args:
        table: Table name (e.g. 'g_demographics', 'g_episodes').
        column: Column name (e.g. 'natio_descr', 'episode_type_ref').
        limit: Max distinct values to return (default 30, max 200).

    Examples:
        list_distinct_values('g_demographics', 'natio_descr')
            → 'Espana', 'América', 'Europa', 'Marruecos', ...
        list_distinct_values('g_episodes', 'episode_type_ref')
            → 'HOSP', 'AMB', 'URG', ...
    """
    if not _safe_identifier(table) or not _safe_identifier(column):
        return "Error: invalid table or column name."
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 30, 200))
    sql = f"SELECT DISTINCT {column} FROM {table} ORDER BY {column} LIMIT {safe_limit}"
    return _run_select(sql)


# ----------------------------
# TOOL 4: execute_query — the main answering query
# ----------------------------
@mcp.tool()
def execute_query(query: str) -> str:
    """Execute the MAIN SELECT or WITH query that answers the user's question.

    Call this AFTER any necessary lookup_ontology, lookup_codes, or list_distinct_values
    calls — never guess literal codes or text values, always look them up first. Only
    SELECT and WITH statements are accepted (INSERT/UPDATE/DELETE/DROP are rejected).

    Returns a JSON list of row dicts. Empty list = no matches.

    Examples of correct use:
        execute_query("SELECT COUNT(DISTINCT patient_ref) FROM g_demographics WHERE sex = 2")
        execute_query("SELECT AVG(result_num) FROM g_labs WHERE lab_sap_ref IN ('LABHG1','LABHG2')")

    Incorrect uses (handled by other tools):
        Understanding a schema-level concept → use lookup_ontology
        Discovering ICD or lab codes → use lookup_codes
        Finding nationality / episode-type strings → use list_distinct_values
        Modifying data → not allowed
    """
    first_word = query.strip().split()[0].upper() if query.strip() else ""
    if first_word not in ("SELECT", "WITH"):
        return "Error: Only SELECT and WITH queries are allowed."
    return _run_select(query)


if __name__ == "__main__":
    mcp.run()