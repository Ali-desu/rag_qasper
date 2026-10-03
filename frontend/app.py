import os

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("Research Paper Assistant")
st.caption("Choose a paper, read its sections, and ask questions about it.")


@st.cache_data(ttl=30)
def load_papers():
    response = requests.get(f"{API_URL}/papers", timeout=30)
    response.raise_for_status()
    return response.json()


try:
    papers = load_papers()
except requests.RequestException as exc:
    st.error(f"Could not load papers from the API at {API_URL}: {exc}")
    st.info("Start the FastAPI server, then reload this page.")
    st.stop()

if not papers:
    st.warning("There are no papers in the database yet.")
    st.stop()

paper_by_id = {paper["paper_id"]: paper for paper in papers}
selected_id = st.selectbox(
    "Choose a paper",
    options=list(paper_by_id),
    format_func=lambda paper_id: paper_by_id[paper_id].get("title") or paper_id,
)

try:
    response = requests.get(f"{API_URL}/papers/{selected_id}", timeout=30)
    response.raise_for_status()
    paper = response.json()
except requests.RequestException as exc:
    st.error(f"Could not load this paper: {exc}")
    st.stop()

st.header(paper.get("title") or "Untitled paper")
with st.expander("Abstract", expanded=True):
    st.write(paper.get("abstract") or "No abstract is available.")

section_blocks = []
for chunk in paper.get("sections", []):
    section_name = chunk["section"]
    if not section_blocks or section_blocks[-1]["name"] != section_name:
        section_blocks.append({"name": section_name, "paragraphs": []})
    section_blocks[-1]["paragraphs"].append(chunk["text"])

st.subheader("Paper")
if not section_blocks:
    st.info("No paper paragraphs were found in the database.")
else:
    section_counts = {}
    for block in section_blocks:
        section_name = block["name"]
        section_counts[section_name] = section_counts.get(section_name, 0) + 1
        repeated_heading = section_counts[section_name] > 1
        display_name = (
            f"{section_name} ({section_counts[section_name]})"
            if repeated_heading
            else section_name
        )
        with st.expander(display_name, expanded=False):
            for paragraph in block["paragraphs"]:
                st.write(paragraph)

st.divider()
st.subheader("Ask about this paper")
with st.form("ask_form"):
    question = st.text_area("Your question", placeholder="What datasets did the authors use?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        try:
            with st.spinner("Reading the paper and preparing an answer…"):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question.strip(), "paper_id": selected_id},
                    timeout=360,
                )
                response.raise_for_status()
                result = response.json()

            st.markdown("### Answer")
            st.write(result.get("answer", "No answer was returned."))
            sections_used = list(dict.fromkeys(result.get("sections_used", [])))
            if sections_used:
                st.caption("Retrieved sections: " + ", ".join(sections_used))
        except requests.RequestException as exc:
            st.error(f"The question request failed: {exc}")
